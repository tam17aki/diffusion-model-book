# -*- coding: utf-8 -*-
"""Sample script for training score-based model

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
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScoreModel(nn.Module):
    """Score-based model."""

    def __init__(self, cfg):
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

    def forward(self, inputs, sigma):
        """Estimate denoising score.

        Args:
            inputs: perturbed sample
            sigma: strength of noise at time t

        Returns:
            outputs: denoising score
        """
        inputs = torch.cat((inputs, sigma), dim=1)
        outputs = self.model(inputs) / sigma
        return outputs

    def get_loss(self, samples, sigmas):
        """Compute loss in SBM.

        Args:
            score_model: score-based model.
            samples: scattered samples from a mixture of Gaussians in 2-d space. [B, 2]
            sigmas: strength of noise at time t. [T]
        """
        # generate B random integers within [0, T]
        # 各時刻での最適化をバッチ方向に並列で行うためのトリック
        times = torch.randint(0, len(sigmas), (samples.shape[0],), device=sigmas.device)

        # 各時刻でのsigmaたち（様々に異なるノイズの強さ）を、バッチ方向に一斉に取り出す
        used_sigmas = sigmas[times].view(
            samples.shape[0], *([1] * len(samples.shape[1:]))
        )

        noise = torch.randn_like(samples) * used_sigmas  # ノイズを生成 [B, 2]
        perturbed = samples + noise  # \tilde{x}; サンプルに擾乱を加える

        target = -1 / (used_sigmas**2) * noise  # 実際にサンプルに加えた「負のノイズ」がtarget

        # 各時刻での擾乱後のサンプルとノイズの強さから、デノイジングスコアを推定
        scores = self.forward(perturbed, used_sigmas)

        # loss計算のためにテンソルを整形
        target = target.view(target.shape[0], -1)  # [B, 1] -> [B]
        scores = scores.view(scores.shape[0], -1)  # [B, 1] -> [B]

        weight = used_sigmas.squeeze(-1) ** 2  # 各時刻で誤差にかける重み
        loss = ((scores - target) ** 2).sum(dim=-1) * weight

        # バッチ（時刻）方向に平均 -> 「sum_{t=1}^{T} w_t * E」のモンテカルロ近似
        # →各時刻は[0, T] からの独立かつ一様なサンプリング（時刻の重複を許したサンプリング）、
        # 時刻に関する「和」を時刻に関する「期待値」で置き換えている
        return loss.mean()

    def sampling(self, sigmas):
        """Perform SBM sampling.

        Args:
            n_sample: number of samples
            score_model: score-based model (neural net)
            sigmas: a set of noise strengths (sigma^t)
            alpha: scale of step size

        Returns:
            x_tk: sample after Langevin Monte Carlo
        """
        n_samples = self.cfg.sampling.n_samples
        n_channels = self.cfg.model.n_channels
        alpha = self.cfg.sampling.alpha
        n_steps = self.cfg.sampling.n_steps
        sigma_last = sigmas[-1]  # noise strength at last step
        x_0 = torch.randn(n_samples, n_channels, device=DEVICE) * sigma_last
        x_tk = x_0
        for time_t in range(len(sigmas) - 1, -1, -1):
            sigma_t = sigmas[time_t]
            alpha_t = alpha * (sigma_t**2) / (sigma_last**2)
            print(f"t:{time_t}, sigma_t:{sigma_t}, alpha_t:{alpha_t}")
            for k in range(n_steps + 1):
                u_k = torch.randn(n_samples, n_channels, device=DEVICE)  # u_k ~ N(0, I)
                if (k == n_steps) and time_t == 0:  # final step
                    u_k[:, :] = 0.0  # no noise is added
                with torch.no_grad():
                    # compute denoising score from perturbed samples and noise
                    score = self.forward(
                        x_tk, torch.ones((n_samples, 1), device=DEVICE) * sigma_t
                    )

                    # sampling via Langevin Monte Carlo
                    x_tk = x_tk + alpha_t * score + torch.sqrt(2 * alpha_t) * u_k

        return x_tk  # x_{0, 0}


def get_model(cfg):
    """Get score-based model."""
    score_model = ScoreModel(cfg).to(DEVICE)
    return score_model


def make_dataset(cfg):
    """Make dataset for training."""
    n_samples = cfg.SBM.n_samples
    sigma = cfg.SBM.sigma

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


def get_sigmas(cfg):
    """Get a set of noise strengths."""
    sigma_begin = cfg.SBM.sigma_begin  # 初期分布
    sigma_end = cfg.SBM.sigma_end  # 最終分布（固定値）
    n_steps = cfg.SBM.n_steps  # 拡散過程の分割ステップ数

    # 対数上で等分割した後にexpでスケールを戻す
    # →各時刻におけるノイズたちを手に入れる
    sigmas = (
        torch.tensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), n_steps))
        )
        .float()
        .to(DEVICE)
    )
    return sigmas


def get_optimizer(cfg: DictConfig, model):
    """Instantiate optimizer."""
    optimizer_class = getattr(optim, cfg.training.optim.optimizer.name)
    optimizer = optimizer_class(
        model.parameters(), **cfg.training.optim.optimizer.params
    )
    return optimizer


def get_lr_scheduler(cfg: DictConfig, optimizer):
    """Instantiate scheduler."""
    lr_scheduler_class = getattr(
        optim.lr_scheduler, cfg.training.optim.lr_scheduler.name
    )
    lr_scheduler = lr_scheduler_class(
        optimizer, **cfg.training.optim.lr_scheduler.params
    )
    return lr_scheduler


def train(dataloader, sigmas, cfg):
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
        loss = model.get_loss(inputs, sigmas)
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
    """Perform training of SBM and sampling."""
    print(OmegaConf.to_yaml(cfg), flush=True)  # dump configuration

    # 訓練用データの準備
    dataset, normalized_samples = make_dataset(cfg)
    dataloader = get_dataloader(dataset, cfg)

    # 真のデータ分布をプロット（ガウス混合分布, 混合数2）
    plot_density(normalized_samples, title="Sample Density")

    # 各時刻におけるノイズたちを手に入れる
    sigmas = get_sigmas(cfg)

    # デノイジングスコアモデルを訓練
    score_model = train(dataloader, sigmas, cfg)

    # 大量のサンプルを一様にばらまいたのちにSBMサンプリングさせる
    samples_pred = score_model.sampling(sigmas)

    # サンプリング結果が2つのモードに集まることを確認 →SBMがうまく動いている
    plot_density(samples_pred, title="Predicted Sample Density")


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="sbm_config")
    main(config)
