"""Portable models with the original checkpoint parameter topology.

SciPlex3 keeps ``flow``, ``flow.control_lora`` and ``decoder`` unchanged.
Task adapters can register their existing native model factories explicitly.
"""
from pathlib import Path
import importlib
import torch
from torch import nn
from .networks.attention_velocity import AttentionCellFlowVelocity
from .networks.state_encoder import SEControlLoRATail
from .networks.expression_decoder import LatentToGeneDecoder
from .flow.integration import rk4_endpoint
from .flow.losses import LossWeights, centered_source_anchor_mse
from .flow.paths import constant_noise_path


class AttentionSystem(nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        self.flow = AttentionCellFlowVelocity(**(model_config or {}))
        self.decoder = LatentToGeneDecoder(2048, 2000, [512, 1024], dropout=0.1,
                                          output_activation="relu")

    def context(self, batch, esm_table):
        return self.flow.prepare_context(
            batch["raw_gene_tokens"], None, batch["drug"], batch["dose"],
            batch["line"], batch["gene_valid"], gene_ids=batch["gene_ids"],
            esm_table=esm_table)


class LoraAttentionSystem(AttentionSystem):
    def __init__(self, model_config, tail_checkpoint, decoder_checkpoint,
                 expected_gene_order_sha256=None, *, loss_weights=None):
        super().__init__(model_config)
        self.model_config = dict(model_config)
        self.loss_weights = LossWeights.from_config(loss_weights)
        with torch.random.fork_rng(devices=[]):
            tail = SEControlLoRATail.load_tail(tail_checkpoint, map_location="cpu")
        if any(float(x) != 0.0 for x in tail.native_attention_dropout):
            raise ValueError("Use the tail exported with the deterministic zero-dropout prefix cache")
        if tail.rank != 8 or tail.alpha != 8.0 or tail.lora_dropout != 0.0:
            raise ValueError("The cell-only adapter requires rank=alpha=8 and dropout=0")
        trainable = {n for n,p in tail.named_parameters() if p.requires_grad}
        expected = {f"linear{i}.lora_{letter}" for i in (1,2) for letter in ("A","B")}
        if trainable != expected:
            raise ValueError("Only four SE tail LoRA matrices may be trainable")
        if any(torch.count_nonzero(p).item() for n,p in tail.named_parameters() if n.endswith("lora_B")):
            raise ValueError("Fresh initialization requires zero-increment LoRA B matrices")
        self.flow.control_lora = tail
        self._control_mean_cache = {}
        self._control_mean_versions = None
        self._control_prefixes = {}
        if decoder_checkpoint is not None:
            payload = torch.load(decoder_checkpoint,map_location="cpu",weights_only=False)
            if payload.get("model_family") != "SciPlex3_true_SE_MSE_MMD_decoder":
                raise ValueError("Expected the independently finetuned true-SE MSE+MMD decoder")
            if expected_gene_order_sha256 and payload.get("gene_order_sha256") != expected_gene_order_sha256:
                raise ValueError("Decoder gene order differs from the active dataset")
            state = payload.get("decoder_state_dict")
            if state is None:
                state = {k[len("decoder."):]:v for k,v in payload["model"].items() if k.startswith("decoder.")}
            self.decoder.load_state_dict(state,strict=True)
        self.decoder.requires_grad_(True)

    @property
    def control_lora(self):
        return self.flow.control_lora

    def bind_control_prefixes(self, prefixes):
        # Runtime owns immutable tensors on the current rank/device; do not
        # checkpoint these data caches or broadcast them through DDP.
        self._control_prefixes = dict(prefixes)
        self._control_mean_cache.clear()
        self._control_mean_versions = None

    def encode_source(self, prefix):
        return self.control_lora(prefix).float()

    @torch.no_grad()
    def current_control_mean(self, line):
        versions = tuple(p._version for p in self.control_lora.parameters() if p.requires_grad)
        if versions != self._control_mean_versions:
            self._control_mean_cache.clear()
            self._control_mean_versions = versions
        if line not in self._control_prefixes:
            raise ValueError("Control prefixes must be bound before computing population means")
        if line not in self._control_mean_cache:
            prefixes = self._control_prefixes[line]
            total = torch.zeros(2048,device=prefixes.device,dtype=torch.float64)
            # This separate context prevents no-grad autocast weight casts from
            # entering the training forward cache. Match the online BF16 tail.
            with torch.autocast(prefixes.device.type,dtype=torch.bfloat16,
                                enabled=prefixes.device.type=="cuda",cache_enabled=False):
                for chunk in prefixes.split(1024):
                    total += self.encode_source(chunk).double().sum(0)
            self._control_mean_cache[line] = (total/len(prefixes)).float()
        return self._control_mean_cache[line]

    def forward(self,batch,esm_table,time_points=10, *, return_loss_tensors=False):
        pairs = int(batch["pairs_per_condition"])
        groups = len(batch["condition_lines"])
        if pairs < 2 or batch["source_prefix"].shape[0] != pairs*groups:
            raise ValueError("Grouped batches must preserve condition-local pair counts")
        source = self.encode_source(batch["source_prefix"])
        target = batch["target"].float()
        context = self.context(batch,esm_table)
        t = batch["time"]
        state = constant_noise_path(source, target, t, batch["noise"])
        velocity = self.flow(state,t,source,context).float()
        desired = target-source
        p = velocity.reshape(groups,pairs,-1)
        y = desired.reshape(groups,pairs,-1)
        pm,tm = p.mean(1),y.mean(1)
        residual_flow = ((p-pm[:,None])-(y-tm[:,None])).square().mean()
        source_means = torch.stack([self.current_control_mean(line) for line in batch["condition_lines"]])
        mean_delta = (batch["target_population_mean"].float()-source_means).detach()
        mean_shift = (pm-mean_delta).square().mean()
        flow_loss = (self.loss_weights.flow_residual*residual_flow
                     + self.loss_weights.flow_mean_shift*mean_shift)
        with torch.no_grad():
            # The source is encoded exactly once and remains fixed throughout
            # RK4. It is never replaced by z_t or re-encoded at later times.
            endpoint = rk4_endpoint(self.flow,source.detach(),context.detach(),time_points)
        # Preserve separate condition-level dropout calls and statistical loss.
        expressions = [self.decoder(part.detach()).float() for part in endpoint.split(pairs)]
        expression = torch.stack(expressions)
        truth = batch["expression"].float().reshape(groups,pairs,-1)
        ep,et = expression.mean(1),truth.mean(1)
        decoder_residual = ((expression-ep[:,None])-(truth-et[:,None])).square().mean()
        decoder_mean = (ep-batch["mean_expression"].float()).square().mean()
        decoder_loss = (self.loss_weights.decoder_residual*decoder_residual
                        + self.loss_weights.decoder_mean*decoder_mean)
        # Keep the default-off arithmetic and dropout trajectory identical to
        # the original four-loss recipe. Optional branches add no parameters.
        total_loss = flow_loss+decoder_loss
        source_anchor = source.new_zeros(())
        decoder_true_se = source.new_zeros(())
        optional_terms = {}
        if self.loss_weights.source_anchor > 0:
            frozen_source = batch["source"].float()
            if frozen_source.shape != source.shape:
                raise ValueError("Frozen source anchors must match the paired LoRA source rows")
            source_anchor = centered_source_anchor_mse(
                source.reshape(groups,pairs,-1),
                frozen_source.reshape(groups,pairs,-1))
            weighted_anchor = self.loss_weights.source_anchor*source_anchor
            total_loss = total_loss+weighted_anchor
            optional_terms["source_anchor_weighted"] = weighted_anchor
        if self.loss_weights.decoder_true_se > 0:
            # This extra decoder call is strictly optional. True treated SE
            # embeddings are training supervision, never inference endpoints.
            true_se_expression = self.decoder(target.detach()).float()
            decoder_true_se = (true_se_expression-truth.detach().reshape_as(true_se_expression)).square().mean()
            weighted_true_se = self.loss_weights.decoder_true_se*decoder_true_se
            total_loss = total_loss+weighted_true_se
            optional_terms["decoder_true_se_weighted"] = weighted_true_se
        parts = {
            "residual_flow_loss":residual_flow.detach(),"mean_shift_loss":mean_shift.detach(),
            "weighted_mean_shift_loss":self.loss_weights.flow_mean_shift*mean_shift.detach(),"flow_loss":flow_loss.detach(),
            "decoder_residual_loss":decoder_residual.detach(),"decoder_mean_loss":decoder_mean.detach(),
            "weighted_decoder_mean_loss":self.loss_weights.decoder_mean*decoder_mean.detach(),"decoder_loss":decoder_loss.detach(),
            "source_anchor_loss":source_anchor.detach(),
            "weighted_source_anchor_loss":self.loss_weights.source_anchor*source_anchor.detach(),
            "decoder_true_se_loss":decoder_true_se.detach(),
            "weighted_decoder_true_se_loss":self.loss_weights.decoder_true_se*decoder_true_se.detach(),
            "total_loss":total_loss.detach(),
        }
        if return_loss_tensors:
            return total_loss, parts, {
                "flow_residual": self.loss_weights.flow_residual*residual_flow,
                "flow_mean_weighted": self.loss_weights.flow_mean_shift*mean_shift,
                "decoder_loss": decoder_loss,
                **optional_terms,
            }
        return total_loss, parts

    @torch.no_grad()
    def predict(self,batch,esm_table,time_points=80):
        source = self.encode_source(batch["source_prefix"])
        context = self.context(batch,esm_table)
        endpoint = rk4_endpoint(self.flow,source,context,time_points)
        return endpoint,self.decoder(endpoint).float()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, *, tail_checkpoint=None,
                        decoder_checkpoint=None, map_location="cpu", loss_weights=None):
        """Restore a full historical model strictly; supply a portable tail path.

        The full model checkpoint already contains decoder parameters, so a
        separate decoder pretraining file is optional during restoration.
        Optimizer/RNG restoration belongs to the trainer.
        """
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if payload.get("schema") != "scratch_attention_cell_lora_training_v1":
            raise ValueError("Unsupported SciPlex3 checkpoint schema")
        tail_checkpoint = tail_checkpoint or payload["config"]["tail_checkpoint"]
        if not Path(tail_checkpoint).is_file():
            raise FileNotFoundError("Supply tail_checkpoint pointing to the original SE tail artifact")
        with torch.random.fork_rng(devices=[]):
            model = cls(payload["model_config"], tail_checkpoint, decoder_checkpoint,
                        loss_weights=loss_weights or payload.get("loss_weights"))
        model.load_state_dict(payload["model"], strict=True)
        return model


SciPlex3Model = LoraAttentionSystem
_TASK_MODELS = {"sciplex3": SciPlex3Model}
_TASK_MODULES = {
    "norman19": "devflow.runtime.norman",
    "norman": "devflow.runtime.norman",
    "combosciplex": "devflow.runtime.combosciplex",
}


def register_task_model(task, factory):
    """Register an explicit task adapter without coupling native architectures."""
    name = str(task).lower()
    if name in _TASK_MODELS and _TASK_MODELS[name] is not factory:
        raise ValueError(f"Model factory already registered for {name}")
    if not callable(factory):
        raise TypeError("Model factory must be callable")
    _TASK_MODELS[name] = factory
    return factory


def _task_factory(task):
    name = str(task).lower()
    if name not in _TASK_MODELS and name in _TASK_MODULES:
        importlib.import_module(_TASK_MODULES[name])
    if name not in _TASK_MODELS:
        raise ValueError(f"Unknown model task {task!r}; register its native adapter first")
    return _TASK_MODELS[name]


def build_model(task, **kwargs):
    return _task_factory(task)(**kwargs)


class DeVFlow:
    """Task model factories returning native modules without extra key prefixes."""

    @staticmethod
    def build(task, **kwargs):
        return build_model(task, **kwargs)

    @staticmethod
    def from_checkpoint(task, checkpoint_path, **kwargs):
        factory = _task_factory(task)
        loader = getattr(factory, "from_checkpoint", None)
        if loader is None:
            raise TypeError(f"Model factory for {task!r} has no checkpoint loader")
        return loader(checkpoint_path, **kwargs)
