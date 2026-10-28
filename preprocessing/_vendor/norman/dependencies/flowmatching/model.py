"""Conditional expression-space velocity field and ODE solvers."""

import torch
from torchdiffeq import odeint
from torch import nn

from .time_adapter import TimestepEmbedder
from .losses import median_sigmas, mmd2_unbiased_multi_sigma


class CategoricalDrugEmbedding(nn.Embedding):
    """Fallback condition table for processed drug IDs.

    A molecular/knowledge encoder can replace this table without changing the
    velocity-field contract. Dose is intentionally absent from this module.
    """

    def __init__(self, num_drugs, embedding_dim=256):
        super().__init__(num_drugs, embedding_dim)


class ConditionalExpressionVelocity(nn.Module):
    """Predict expression-space velocity conditioned on cell latent and drug."""

    def __init__(self, expression_dim, cell_latent_dim=2048, drug_dim=256,
                 hidden_dim=1024, time_dim=256, depth=4,
                 cell_condition_is_encoded=False):
        super().__init__()
        self.time_encoder = TimestepEmbedder(time_dim)
        if cell_condition_is_encoded:
            if cell_latent_dim != hidden_dim:
                raise ValueError("encoded cell condition dimension must equal hidden_dim")
            self.cell_encoder = nn.LayerNorm(cell_latent_dim)
        else:
            self.cell_encoder = nn.Sequential(
                nn.LayerNorm(cell_latent_dim), nn.Linear(cell_latent_dim, hidden_dim), nn.SiLU()
            )
        self.drug_encoder = nn.Sequential(nn.LayerNorm(drug_dim), nn.Linear(drug_dim, hidden_dim), nn.SiLU())
        dims = [expression_dim + hidden_dim + hidden_dim + time_dim] + [hidden_dim] * (depth - 1) + [expression_dim]
        layers = []
        for left, right in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(left, right), nn.SiLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.velocity = nn.Sequential(*layers)

    def forward(self, x_t, cell_latent, drug_embedding, t):
        if x_t.ndim != 2:
            raise ValueError(f"x_t must be [N, expression_dim], got {tuple(x_t.shape)}")
        if cell_latent.shape[0] != x_t.shape[0] or drug_embedding.shape[0] != x_t.shape[0]:
            raise ValueError("state and condition batch sizes must match")
        condition = torch.cat([
            self.cell_encoder(cell_latent),
            self.drug_encoder(drug_embedding),
            self.time_encoder(t),
        ], dim=-1)
        return self.velocity(torch.cat([x_t, condition], dim=-1))


class FlowMatchingSystem(nn.Module):
    """Trainable control fusion and velocity field under one DDP boundary."""

    def __init__(self, expression_dim=2000, hidden_dim=1024, time_dim=256, depth=4,
                 fusion_hidden_dim=2048, fusion_dropout=0.1):
        super().__init__()
        from .upstream import ControlConditionFusion

        self.condition_fusion = ControlConditionFusion(
            hidden_dim=fusion_hidden_dim,
            output_dim=hidden_dim,
            dropout=fusion_dropout,
        )
        self.velocity = ConditionalExpressionVelocity(
            expression_dim=expression_dim,
            cell_latent_dim=hidden_dim,
            drug_dim=1024,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            depth=depth,
            cell_condition_is_encoded=True,
        )

    def _conditions(self, control_expression, control_h, control_g, drug_embedding):
        if control_expression.ndim != 3:
            raise ValueError("control_expression must be [B,S,D]")
        batch_size, set_size, expression_dim = control_expression.shape
        h_c = self.condition_fusion(
            control_h.reshape(batch_size * set_size, -1),
            control_g.reshape(batch_size * set_size, -1),
        )
        expanded_drug = drug_embedding[:, None, :].expand(-1, set_size, -1).reshape(
            batch_size * set_size, -1
        )
        return batch_size, set_size, expression_dim, h_c, expanded_drug

    def forward(self, control_expression, control_h, control_g, drug_embedding,
                target, euler_steps=4, objective="endpoint_mmd", mmd_weight=0.5,
                return_components=False):
        batch_size, set_size, expression_dim, h_c, expanded_drug = self._conditions(
            control_expression, control_h, control_g, drug_embedding
        )
        if objective == "cfm_mmd":
            target_flat = target.reshape(batch_size * set_size, expression_dim)
            noise = torch.randn_like(target_flat)
            time = torch.rand(batch_size * set_size, device=target.device, dtype=target.dtype)
            time_column = time[:, None]
            state = (1.0 - time_column) * noise + time_column * target_flat
            target_velocity = target_flat - noise
            predicted_velocity = self.velocity(state, h_c, expanded_drug, time)
            cfm_loss = (predicted_velocity.float() - target_velocity.float()).square().mean()
            endpoint = state + (1.0 - time_column) * predicted_velocity
            endpoint = endpoint.reshape(batch_size, set_size, expression_dim)
            mmd_losses = []
            for index in range(batch_size):
                endpoint_float = endpoint[index].float()
                target_float = target[index].float()
                sigmas = median_sigmas(target_float.detach())
                mmd_losses.append(mmd2_unbiased_multi_sigma(endpoint_float, target_float, sigmas))
            mmd_loss = torch.stack(mmd_losses).mean()
            total_loss = cfm_loss + float(mmd_weight) * mmd_loss
            if return_components:
                return total_loss, endpoint, {
                    "cfm_loss": cfm_loss,
                    "mmd_loss": mmd_loss,
                    "weighted_mmd_loss": float(mmd_weight) * mmd_loss,
                }
            return total_loss, endpoint
        if objective != "endpoint_mmd":
            raise ValueError(f"unknown training objective: {objective}")
        prediction = euler_rollout(
            self.velocity,
            control_expression.reshape(batch_size * set_size, expression_dim),
            h_c,
            expanded_drug,
            steps=euler_steps,
        ).reshape(batch_size, set_size, expression_dim)
        losses = []
        for index in range(batch_size):
            prediction_float = prediction[index].float()
            target_float = target[index].float()
            sigmas = median_sigmas(target_float.detach())
            losses.append(mmd2_unbiased_multi_sigma(prediction_float, target_float, sigmas))
        return torch.stack(losses).mean(), prediction

    def sample(self, control_expression, control_h, control_g, drug_embedding,
               time_points=20, noise=None):
        batch_size, set_size, expression_dim, h_c, expanded_drug = self._conditions(
            control_expression, control_h, control_g, drug_embedding
        )
        if noise is None:
            noise = torch.randn_like(control_expression)
        prediction = rk4_rollout(
            self.velocity,
            noise.reshape(batch_size * set_size, expression_dim),
            h_c,
            expanded_drug,
            time_points=time_points,
        )
        return prediction.reshape(batch_size, set_size, expression_dim)


class ExpressionControlOTFlowMatchingSystem(nn.Module):
    """Real-control OTFM in normalized expression space."""

    def __init__(self, expression_dim=2000, hidden_dim=1024, time_dim=256, depth=4,
                 fusion_hidden_dim=2048, fusion_dropout=0.1):
        super().__init__()
        from .upstream import ControlConditionFusion

        self.condition_fusion = ControlConditionFusion(
            hidden_dim=fusion_hidden_dim,
            output_dim=hidden_dim,
            dropout=fusion_dropout,
        )
        self.velocity = ConditionalExpressionVelocity(
            expression_dim=expression_dim,
            cell_latent_dim=hidden_dim,
            drug_dim=1025,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            depth=depth,
            cell_condition_is_encoded=True,
        )

    @staticmethod
    @torch.no_grad()
    def _pair(control_expression, control_h, control_g, target, epsilon,
              max_iterations, tolerance):
        from .ot_matching import pair_population

        paired = [[], [], [], []]
        for source, source_h, source_g, treated in zip(
            control_expression, control_h, control_g, target
        ):
            source_index, target_index, _ = pair_population(
                source,
                treated,
                n_samples=source.size(0),
                epsilon=epsilon,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            paired[0].append(source.index_select(0, source_index))
            paired[1].append(source_h.index_select(0, source_index))
            paired[2].append(source_g.index_select(0, source_index))
            paired[3].append(treated.index_select(0, target_index))
        return tuple(torch.stack(values) for values in paired)

    def _conditions(self, control_h, control_g, drug_embedding, dose):
        batch_size, set_size = control_h.shape[:2]
        fused = self.condition_fusion(
            control_h.reshape(batch_size * set_size, -1),
            control_g.reshape(batch_size * set_size, -1),
        )
        drug_and_dose = torch.cat((drug_embedding, dose), dim=-1)
        expanded = drug_and_dose[:, None, :].expand(-1, set_size, -1)
        return fused, expanded.reshape(batch_size * set_size, -1)

    def forward(self, control_expression, control_h, control_g, drug_embedding,
                dose, target, path_sigma=1.5, ot_epsilon=1.0,
                ot_max_iterations=2000, ot_tolerance=1e-3,
                return_components=False):
        source, source_h, source_g, paired_target = self._pair(
            control_expression,
            control_h,
            control_g,
            target,
            ot_epsilon,
            ot_max_iterations,
            ot_tolerance,
        )
        batch_size, set_size, expression_dim = source.shape
        fused, condition = self._conditions(source_h, source_g, drug_embedding, dose)
        source_flat = source.reshape(batch_size * set_size, expression_dim)
        target_flat = paired_target.reshape(batch_size * set_size, expression_dim)
        time = torch.rand(
            batch_size * set_size, device=source.device, dtype=source.dtype
        )
        state = (
            (1.0 - time[:, None]) * source_flat
            + time[:, None] * target_flat
            + float(path_sigma) * torch.randn_like(source_flat)
        )
        target_velocity = target_flat - source_flat
        predicted_velocity = self.velocity(state, fused, condition, time)
        loss = (predicted_velocity.float() - target_velocity.float()).square().mean()
        if return_components:
            zero = loss.new_zeros(())
            return loss, predicted_velocity, {
                "cfm_loss": loss,
                "mmd_loss": zero,
                "weighted_mmd_loss": zero,
            }
        return loss, predicted_velocity

    def sample(self, control_expression, control_h, control_g, drug_embedding,
               dose, time_points=20):
        batch_size, set_size, expression_dim = control_expression.shape
        fused, condition = self._conditions(control_h, control_g, drug_embedding, dose)
        prediction = rk4_rollout(
            self.velocity,
            control_expression.reshape(batch_size * set_size, expression_dim),
            fused,
            condition,
            time_points=time_points,
        )
        return prediction.reshape(batch_size, set_size, expression_dim)


def euler_rollout(velocity, x0, cell_latent, drug_embedding, steps=4):
    """Integrate from flow time 0 to 1 with differentiable uniform Euler steps."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    x = x0
    dt = 1.0 / steps
    for index in range(steps):
        t = torch.full((x.size(0),), index * dt, device=x.device, dtype=x.dtype)
        x = x + dt * velocity(x, cell_latent, drug_embedding, t)
    return x


def rk4_rollout(velocity, x0, cell_latent, drug_embedding, time_points=20, steps=None):
    """Integrate from flow time 0 to 1 on a fixed grid with classical RK4."""
    if steps is not None:
        time_points = int(steps) + 1
    if time_points < 2:
        raise ValueError("time_points must be >= 2")
    x = x0
    dt = 1.0 / (time_points - 1)
    for index in range(time_points - 1):
        time = index * dt
        t0 = torch.full((x.size(0),), time, device=x.device, dtype=x.dtype)
        tm = torch.full((x.size(0),), time + 0.5 * dt, device=x.device, dtype=x.dtype)
        t1 = torch.full((x.size(0),), time + dt, device=x.device, dtype=x.dtype)
        k1 = velocity(x, cell_latent, drug_embedding, t0)
        k2 = velocity(x + 0.5 * dt * k1, cell_latent, drug_embedding, tm)
        k3 = velocity(x + 0.5 * dt * k2, cell_latent, drug_embedding, tm)
        k4 = velocity(x + dt * k3, cell_latent, drug_embedding, t1)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def adaptive_rollout(velocity, x0, cell_latent, drug_embedding,
                     rtol=1e-5, atol=1e-5, method="dopri5"):
    """Integrate the learned velocity field with an adaptive-step ODE solver."""
    if x0.ndim != 2:
        raise ValueError("x0 must be [N,D]")

    class RHS(torch.nn.Module):
        def forward(self, time, state):
            t = time.to(device=state.device, dtype=state.dtype).reshape(1).expand(state.size(0))
            return velocity(state, cell_latent, drug_embedding, t)

    times = torch.tensor([0.0, 1.0], device=x0.device, dtype=x0.dtype)
    return odeint(RHS(), x0, times, method=method, rtol=float(rtol), atol=float(atol))[-1]


def endpoint_mmd_loss(velocity, x0, cell_latent, drug_embedding, target, steps=4):
    """Roll out a condition and compare only its endpoint distribution."""
    prediction = euler_rollout(velocity, x0, cell_latent, drug_embedding, steps=steps)
    sigmas = median_sigmas(target.detach())
    return mmd2_unbiased_multi_sigma(prediction, target, sigmas=sigmas), prediction
