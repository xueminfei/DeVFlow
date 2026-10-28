"""Gradient routes and actual updates for the existing model components."""
import math
import torch


def tracked_parameters(model):
    tracked = {}
    for name, parameter in model.named_parameters():
        if name.startswith("flow.control_lora.") and (name.endswith("lora_A") or name.endswith("lora_B")):
            tracked[name] = parameter
        elif parameter.requires_grad and any(x in name for x in ("time_modulation.1.weight", "time_embedder", "cell_projector", "gene_projector", "project_out")) and name.endswith("weight"):
            tracked[name] = parameter
    # Include explicit representatives even when native layer names change.
    for prefix in ("flow.", "decoder."):
        for name, parameter in model.named_parameters():
            if name.startswith(prefix) and parameter.requires_grad:
                tracked.setdefault(name, parameter)
                break
    return tracked


def gradient_snapshot(parameters):
    return {name: {"requires_grad": p.requires_grad, "has_gradient": p.grad is not None,
                   "gradient_norm": float(p.grad.float().norm()) if p.grad is not None else 0.0}
            for name, p in parameters.items()}


def route_probe(model, batch, esm, time_points):
    # Check every trainable parameter, including layers not selected as update
    # representatives. Otherwise an auxiliary route could leak unnoticed.
    tracked = {n: p for n, p in model.named_parameters() if p.requires_grad}
    device_type = next(model.parameters()).device.type
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
        total, parts, terms = model(batch, esm, time_points, return_loss_tensors=True)
    gradients = {}
    for term, value in terms.items():
        # A source anchor with explicitly frozen LoRA is a detached scalar.
        # Recording absent routes preserves this valid ablation setting.
        derivatives = (torch.autograd.grad(value, list(tracked.values()), retain_graph=True, allow_unused=True)
                       if value.requires_grad else (None,)*len(tracked))
        gradients[term] = dict(zip(tracked, derivatives))
    rows = {}
    for name in tracked:
        row = {}
        for term, mapping in gradients.items():
            gradient = mapping[name]
            row[term] = None if gradient is None else float(gradient.float().norm())
        residual = gradients["flow_residual"][name]
        mean = gradients["flow_mean_weighted"][name]
        if residual is not None and mean is not None:
            denominator = residual.float().norm() * mean.float().norm()
            row["flow_term_cosine"] = float((residual.float() * mean.float()).sum() / denominator) if denominator > 0 else None
        if name.startswith("flow.") and gradients["decoder_loss"][name] is not None:
            raise RuntimeError(f"Expression gradient unexpectedly reached {name}")
        if ("source_anchor_weighted" in gradients
                and not name.startswith("flow.control_lora.")
                and gradients["source_anchor_weighted"][name] is not None):
            raise RuntimeError(f"Source anchor gradient unexpectedly reached {name}")
        if ("decoder_true_se_weighted" in gradients
                and not name.startswith("decoder.")
                and gradients["decoder_true_se_weighted"][name] is not None):
            raise RuntimeError(f"True-SE reconstruction gradient unexpectedly reached {name}")
        rows[name] = row
    optional_routes = {}
    for term, prefix in (("source_anchor_weighted", "flow.control_lora."),
                         ("decoder_true_se_weighted", "decoder.")):
        if term in gradients:
            expected = [name for name in tracked if name.startswith(prefix)]
            connected = [name for name in expected if gradients[term][name] is not None]
            if expected and not connected:
                raise RuntimeError(f"Enabled {term} is disconnected from {prefix}")
            optional_routes[term] = {
                "allowed_prefix": prefix,
                "trainable_parameters": len(expected),
                "connected_parameters": len(connected),
                "status": "passed" if expected else "frozen",
            }
    return {"status": "passed", "parameters": rows,
            "losses": {key: float(value) for key, value in parts.items()},
            "expression_gradient_to_flow": False,
            "optional_routes": optional_routes}
