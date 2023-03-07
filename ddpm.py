# -*- coding: utf-8 -*-
"""Sample script for training Denoising Diffusion Probabilistic Model

Copyright (C) 2023 by Shuji SUZUKI
Copyright (C) 2023 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch import optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model."""

    def __init__(self, cfg):
        """Initialize class."""
        super().__init__()
        self.cfg = cfg
        n_channels = cfg.model.n_channels
        hidden_dim = cfg.model.h_dim
        layers = []
        layers += [nn.Sequential(nn.Linear(n_channels + 1, hidden_dim), nn.ReLU())]
        layers += [
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(self.cfg.model.n_hidden)
        ]
        layers += [nn.Linear(hidden_dim, n_channels)]

        self.model = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()

    def forward(self, inputs, t):
        """Estimate perturbation noise.

        Args:
            inputs: perturbed sample
            t: time step

        Returns:
            outputs: perturbation noise
        """
        inputs = torch.cat((inputs, t.unsqueeze(-1)), dim=1)
        outputs = self.model(inputs)
        return outputs

    def get_loss(self, inputs, alpha_bars, beta_bars):
        """Compute loss in DDPM.

        Args:
            inputs: scattered samples from a mixture of Gaussians [B, C]
            alphaars: a set of noise strengths [T]
            beta_bars: a set of noise strengths [T]

        Returns:
            loss: DDPM MSE-loss
        """
        # generate B random integers within [0, T]
        t = torch.randint(0, len(inputs), (inputs.shape[0],), device=DEVICE)
        noise = torch.randn_like(inputs)
        perturbed = torch.sqrt(alpha_bars[t]).unsqueeze(-1) * inputs
        perturbed += torch.sqrt(beta_bars[t]).unsqueeze(-1) * noise
        noise_pred = self.forward(perturbed, t)
        loss = self.cfg.model.loss_weight * self.criterion(noise_pred, noise)
        return loss

    def sampling(self, alphas, betas, beta_bars):
        """Perform DDPM sampling.

        Args:
            alphas: a set of noise schedules [T]
            betas: a set of noise schedules [T]
            beta_bars: a set of noise strengths [T]

        Returns:
            xt: sample from reverse diffusion process [N, C]
        """
        n_samples = self.cfg.sampling.n_samples
        n_channels = self.cfg.model.n_channels
        xt = torch.randn(n_samples, n_channels, device=DEVICE)
        T = len(alphas)
        for t in range(T - 1, -1, -1):
            print(f"t:{t}")
            ut = torch.randn(n_samples, n_channels, device=DEVICE)
            if t == 0:
                ut[:, :] = 0.0
            with torch.no_grad():
                noise_pred = self.forward(
                    xt, t * torch.ones(n_samples, dtype=xt.dtype, device=DEVICE)
                )
                sigma_t = torch.sqrt(betas[t])
                xt = xt - betas[t] / torch.sqrt(beta_bars[t]) * noise_pred
                xt *= 1 / torch.sqrt(alphas[t])
                xt += sigma_t * ut
        return xt


def get_model(cfg):
    """Get score-based model."""
    score_model = DDPM(cfg).to(DEVICE)
    return score_model


def make_dataset(cfg):
    """Make dataset for training."""
    n_samples = cfg.DDPM.n_samples
    sigma = cfg.DDPM.sigma

    dist0 = torch.distributions.MultivariateNormal(
        torch.tensor([-2, -2], dtype=torch.float, device=DEVICE),
        sigma * torch.eye(2, dtype=torch.float, device=DEVICE),
    )
    samples0 = dist0.sample(torch.Size([n_samples // 2]))

    dist1 = torch.distributions.MultivariateNormal(
        torch.tensor([2, 2], dtype=torch.float, device=DEVICE),
        sigma * torch.eye(2, dtype=torch.float, device=DEVICE),
    )
    samples1 = dist1.sample(torch.Size([n_samples // 2]))
    samples = torch.vstack((samples0, samples1))  # 各分布から等確率で生成

    mean = torch.mean(samples, dim=0, keepdim=True)
    std = torch.std(samples, dim=0, keepdim=True)
    normalized_samples = (samples - mean) / std

    dataset = torch.utils.data.TensorDataset((normalized_samples))
    return dataset, normalized_samples


def get_dataloader(dataset, cfg):
    """Get dataloader from dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0
    )
    return dataloader


def get_noise_schedules(cfg):
    """Get noise schedules (alpha & beta)."""
    beta_start = cfg.DDPM.beta_start  # 初期分布 β_0
    beta_end = cfg.DDPM.beta_end  # 最終分布 β_T
    T = cfg.DDPM.T  # 拡散過程の分割ステップ数

    # ノイズスケジュールたちを決定
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32, device=DEVICE)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, axis=0)
    beta_bars = 1.0 - alpha_bars
    return alphas, betas, alpha_bars, beta_bars


def get_optimizer(cfg: DictConfig, model):
    """Instantiate optimizer."""
    optimizer_class = getattr(optim, cfg.training.optim.optimizer.name)
    optimizer = optimizer_class(
        model.parameters(), **cfg.training.optim.optimizer.params
    )
    return optimizer


def get_lr_scheduler(cfg: DictConfig, optimizer):
    """Instantiate scheduler form optimizer."""
    lr_scheduler_class = getattr(
        optim.lr_scheduler, cfg.training.optim.lr_scheduler.name
    )
    lr_scheduler = lr_scheduler_class(
        optimizer, **cfg.training.optim.lr_scheduler.params
    )
    return lr_scheduler


def training(cfg: DictConfig, dataloader, alpha_bars, beta_bars):
    """Perform training."""
    dataloader_iter = iter(dataloader)
    model = get_model(cfg)
    optimizer = get_optimizer(cfg, model)
    if cfg.training.use_scheduler:
        lr_scheduler = get_lr_scheduler(cfg, optimizer)
    for i in range(cfg.training.n_steps):
        try:
            inputs = next(dataloader_iter)[0]
        except StopIteration:
            dataloader_iter = iter(dataloader)
            inputs = next(dataloader_iter)[0]
        inputs = inputs.to(DEVICE)

        optimizer.zero_grad()
        loss = model.get_loss(inputs, alpha_bars, beta_bars)
        loss.backward()
        optimizer.step()
        if cfg.training.use_scheduler:
            lr_scheduler.step()
        if (i % cfg.training.log_step) == 0:
            print(f"{i} steps loss:{loss}")

    return model


def plot_density(samples, title="Sample Density"):
    """Plot density via histograms

    Args:
        samples: 2-D Tensors [N, 2]
    """
    plot_samples = samples.cpu().numpy()
    plt.hist2d(
        plot_samples[:, 0],
        plot_samples[:, 1],
        range=((-2, 2), (-2, 2)),
        cmap="viridis",
        rasterized=False,
        bins=100,
        density=True,
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.title(title)
    plt.show()


def main(cfg: DictConfig):
    """Perform training raiD and sampling.DPM t"""
    print(OmegaConf.to_yaml(cfg), flush=True)  # dump configuration

    # 訓練用データの準備
    dataset, normalized_samples = make_dataset(cfg)
    dataloader = get_dataloader(dataset, cfg)

    # 真のデータ分布をプロット（ガウス混合分布, 混合数2）
    plot_density(normalized_samples, title="Sample Density")

    # 各時刻におけるノイズたちを手に入れる
    alphas, betas, alpha_bars, beta_bars = get_noise_schedules(cfg)

    # DDPMを訓練
    model = training(cfg, dataloader, alpha_bars, beta_bars)

    # 大量のサンプルを一様にばらまいたのちにDDPMサンプリングさせる
    samples_pred = model.sampling(alphas=alphas, betas=betas, beta_bars=beta_bars)

    # サンプリング結果が2つのモードに集まることを確認 →DDPMがうまく動いている
    plot_density(samples_pred, title="Predicted Sample Density")


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="ddpm_config")
    main(config)
