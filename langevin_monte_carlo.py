import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def log_p_dist(x):
    dist0 = torch.distributions.MultivariateNormal(torch.tensor([-2, -2]), torch.eye(2))
    p0 = torch.exp(dist0.log_prob(x))
    dist1 = torch.distributions.MultivariateNormal(torch.tensor([2, 2]), torch.eye(2))
    p1 = torch.exp(dist1.log_prob(x))
    return torch.log((p0 + p1) / 2.0)


def langevin_monte_carlo_algorithm(log_p_dist, n_samples=100000, alpha=0.001, K=1000):
    x0 = torch.randn(n_samples, 2)  # N(0, 1)
    x_k = x0
    for i in tqdm(range(K + 1)):
        x_k.requires_grad_()
        log_p = log_p_dist(x_k)
        score = torch.autograd.grad(log_p.sum(), x_k)[0]
        with torch.no_grad():
            u_k = torch.randn(n_samples, 2)
            x_k = x_k + alpha * score + np.sqrt(2 * alpha) * u_k
    return x_k


def main():
    """main function"""
    r = np.linspace(-5, 5, 1000)
    x0, x1 = np.meshgrid(r, r)
    x = torch.tensor(np.vstack([x0.flatten(), x1.flatten()]).T)

    log_p = log_p_dist(x)
    p = torch.exp(log_p)
    plt.pcolormesh(x0, x1, p.reshape(x0.shape), cmap="viridis")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.show()

    samples_1em1 = (
        langevin_monte_carlo_algorithm(log_p_dist, alpha=1e-1).detach().numpy()
    )
    plt.hist2d(
        samples_1em1[:, 0],
        samples_1em1[:, 1],
        range=((-5, 5), (-5, 5)),
        cmap="viridis",
        rasterized=False,
        bins=100,
        density=True,
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.show()

    samples_1em5 = (
        langevin_monte_carlo_algorithm(log_p_dist, alpha=1e-5).detach().numpy()
    )
    plt.hist2d(
        samples_1em5[:, 0],
        samples_1em5[:, 1],
        range=((-5, 5), (-5, 5)),
        cmap="viridis",
        rasterized=False,
        bins=100,
        density=True,
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.show()


if __name__ == "__main__":
    main()
