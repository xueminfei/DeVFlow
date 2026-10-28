"""The original constant-noise conditional interpolation path."""


def constant_noise_path(source, target, time, noise, sigma=0.01):
    """Return the training state, preserving the original operation order."""
    return (1 - time[:, None]) * source + time[:, None] * target + sigma * noise


def constant_velocity_target(source, target):
    return target - source
