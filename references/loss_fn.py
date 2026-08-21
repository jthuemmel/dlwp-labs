import math

import torch


def get_statistics(prediction: torch.Tensor, dim: int = 1, mode: str = "ensemble", epsilon: float = 1e-9):
    if mode == "ensemble":
        mu, sigma = prediction.mean(dim=dim), prediction.std(dim=dim)
    elif mode == "parametric":
        mu, sigma = prediction.split(1, dim=dim)
        mu, sigma = mu.squeeze(dim=dim), sigma.squeeze(dim=dim)
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")
    return mu, sigma + epsilon


def f_mse(observation: torch.Tensor, prediction: torch.Tensor, **kwargs) -> torch.Tensor:
    return (prediction - observation) ** 2


def f_mae(observation: torch.Tensor, prediction: torch.Tensor, **kwargs) -> torch.Tensor:
    return (prediction - observation).abs()


def f_gaussian_crps(observation: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
    sqrtPi, sqrtTwo = math.sqrt(math.pi), math.sqrt(2)
    sigma = sigma.clamp(min=1e-6)
    z = (observation - mu) / sigma
    phi = torch.exp(-z ** 2 / 2) / (sqrtTwo * sqrtPi)
    return sigma * (z * torch.erf(z / sqrtTwo) + 2 * phi - 1 / sqrtPi)


def f_kernel_crps(observation: torch.Tensor, ensemble: torch.Tensor, fair: bool = False, **kwargs) -> torch.Tensor:
    n_member = ensemble.shape[-1]
    coef = -1 / (n_member * (n_member - 1)) if fair else -1 / (n_member ** 2)
    absolute_error = torch.mean((ensemble - observation[..., None]).abs(), dim=-1)
    ens_var = torch.zeros(size=ensemble.shape[:-1], device=ensemble.device)
    for i in range(n_member):
        ens_var += torch.sum(torch.abs(ensemble[..., i, None] - ensemble[..., i + 1:]), dim=-1)
    return absolute_error + coef * ens_var
