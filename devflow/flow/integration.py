"""Classical RK4 with one fixed original source per rollout."""
import torch

def rk4_endpoint(flow, source, context, time_points):
    """time_points includes both endpoints; 4*(time_points-1) network calls."""
    if int(time_points) < 2:
        raise ValueError("RK4 needs at least two time points")
    field = flow.bind_source(source.float(), context)
    state = source.float().clone()
    dt = 1.0 / (int(time_points) - 1)
    for index in range(int(time_points) - 1):
        t = index * dt
        k1 = field(state, t).float()
        k2 = field(state + (0.5 * dt) * k1, t + 0.5 * dt).float()
        k3 = field(state + (0.5 * dt) * k2, t + 0.5 * dt).float()
        k4 = field(state + dt * k3, t + dt).float()
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return state

