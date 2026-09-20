import torch
import torch.nn.functional as F


def vicreg_loss(z, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0,
                target_std=0.5, eps=1e-4,
                target_rate=None, rate_coeff=0.0):
    N, V, D = z.shape
    z = z.float()

    z_mean = z.mean(dim=1, keepdim=True)
    sim_loss = ((z - z_mean) ** 2).mean()

    z_flat = z.reshape(N * V, D)

    std = torch.sqrt(z_flat.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(target_std - std).mean()

    z_c = z_flat - z_flat.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / (N * V - 1)
    off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
    cov_loss = (off_diag ** 2).sum() / D

    total = sim_coeff * sim_loss + std_coeff * var_loss + cov_coeff * cov_loss

    rate_loss = z_flat.new_zeros(())
    if target_rate is not None and rate_coeff > 0:
        p = z_flat.new_tensor(float(target_rate))
        r = z_flat.mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
        rate_loss = (p * (p / r).log()
                     + (1.0 - p) * ((1.0 - p) / (1.0 - r)).log()).mean()
        total = total + rate_coeff * rate_loss

    return total, sim_loss, var_loss, cov_loss, rate_loss