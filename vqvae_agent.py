"""
VQ-VAE World Model Agent adapted for 64x64 highway images.
Based on "Smaller World Models for RL" (Robine et al., arXiv:2010.05767v2).

This module provides a VQVAEAgent class that can be used as an alternative
to the DreamerV3 agent for training on highway environments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# -----------------------------
# Utilities
# -----------------------------

def to_torch(x, device: torch.device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for 2D feature maps."""
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


def continuous_bernoulli_nll(x: torch.Tensor, logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Negative log-likelihood for Continuous Bernoulli distribution."""
    x = torch.clamp(x, 0.0, 1.0)
    lam = torch.sigmoid(logits)
    lam = torch.clamp(lam, eps, 1.0 - eps)

    bce = F.binary_cross_entropy(lam, x, reduction='none')
    t = 1.0 - 2.0 * lam
    abs_t = torch.abs(t)

    small = abs_t < 1e-3
    t2 = t * t
    logC_series = math.log(2.0) + (t2 / 3.0) + (2.0 * t2 * t2 / 15.0)
    atanh_t = 0.5 * (torch.log1p(t) - torch.log1p(-t))
    logC_general = torch.log(2.0 * atanh_t / t)
    logC = torch.where(small, logC_series, logC_general)

    nll = bce - logC
    return nll.mean()


# -----------------------------
# VQ-VAE for 64x64 images -> 8x8 codes
# -----------------------------

class Encoder64(nn.Module):
    """
    Encoder producing 8x8xD latents from 64x64x3 observations.
    Uses stride-2 convs: 64->32->16->8.
    """
    def __init__(self, in_channels: int = 3, hidden: int = 128, embedding_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 4, stride=2, padding=1),   # 64->32
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1),        # 32->16
            nn.ReLU(),
            nn.Conv2d(hidden, embedding_dim, 4, stride=2, padding=1)  # 16->8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder64(nn.Module):
    """Decoder from 8x8xD back to 64x64x3 logits."""
    def __init__(self, embedding_dim: int = 32, hidden: int = 128, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden, 4, stride=2, padding=1),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1),         # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, out_channels, 4, stride=2, padding=1)    # 32->64
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VectorQuantizer(nn.Module):
    """Standard VQ layer with straight-through estimator."""
    def __init__(self, num_embeddings: int = 128, embedding_dim: int = 32, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, d, h, w = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous().view(-1, d)

        emb = self.embeddings.weight
        dist = (
            (z ** 2).sum(dim=1, keepdim=True)
            - 2.0 * z @ emb.t()
            + (emb ** 2).sum(dim=1, keepdim=True).t()
        )

        indices = torch.argmin(dist, dim=1)
        z_q = self.embeddings(indices).view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        z_q_st = z_e + (z_q - z_e).detach()
        indices = indices.view(b, h, w)
        return z_q_st, vq_loss, indices


class VQVAE64(nn.Module):
    """VQ-VAE for 64x64 RGB images."""
    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 32,
        num_embeddings: int = 128,
        commitment_cost: float = 0.25,
        hidden: int = 128,
    ):
        super().__init__()
        self.latent_size = 8  # 64/8 = 8
        self.encoder = Encoder64(in_channels=in_channels, hidden=hidden, embedding_dim=embedding_dim)
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder64(embedding_dim=embedding_dim, hidden=hidden, out_channels=in_channels)

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x)
        _, _, indices = self.vq(z_e)
        return indices

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        logits = self.decoder(z_q)
        return {"logits": logits, "vq_loss": vq_loss, "indices": indices, "z_q": z_q}


# -----------------------------
# ConvLSTM Dynamics Model for 8x8 latents
# -----------------------------

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM Cell with Layer Normalization."""
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)
        self.ln = ChannelLayerNorm2d(4 * hidden_channels)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.ln(self.conv(combined))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch_size: int, spatial: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = spatial
        h0 = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        c0 = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        return h0, c0


class DynamicsModel64(nn.Module):
    """
    Predicts next code indices distribution and reward from current embeddings + action.
    For 64x64 images -> 8x8 latents.
    """
    def __init__(
        self,
        num_embeddings: int = 128,
        embedding_dim: int = 32,
        num_actions: int = 5,
        hidden_channels: int = 256,
        latent_size: int = 8,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        self.hidden_channels = hidden_channels
        self.latent_size = latent_size

        self.embeddings: Optional[nn.Embedding] = None

        self.convlstm1 = ConvLSTMCell(embedding_dim + num_actions, hidden_channels)
        self.convlstm2 = ConvLSTMCell(hidden_channels + num_actions, hidden_channels)

        self.latent_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 256, kernel_size=3, padding=1),
            ChannelLayerNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, num_embeddings, kernel_size=1)
        )

        # Reward head for continuous reward prediction
        self.reward_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
            ChannelLayerNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32 * latent_size * latent_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)  # Continuous reward prediction
        )

    def set_embeddings(self, embeddings: nn.Embedding):
        self.embeddings = embeddings

    def _embed_indices(self, idx: torch.Tensor) -> torch.Tensor:
        if self.embeddings is None:
            raise RuntimeError("Embeddings not set.")
        z = self.embeddings(idx)
        return z.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        indices: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        device = indices.device
        b = indices.shape[0]
        ls = self.latent_size

        z = self._embed_indices(indices)

        # One-hot action planes
        a_onehot = F.one_hot(actions.long(), num_classes=self.num_actions).float().to(device)
        a_planes = a_onehot.view(b, self.num_actions, 1, 1).expand(b, self.num_actions, ls, ls)

        x1 = torch.cat([z, a_planes], dim=1)

        if state is None:
            s1 = self.convlstm1.init_state(b, (ls, ls), device)
            s2 = self.convlstm2.init_state(b, (ls, ls), device)
        else:
            s1, s2 = state

        h1, c1 = self.convlstm1(x1, s1)
        x2 = torch.cat([h1, a_planes], dim=1)
        h2, c2 = self.convlstm2(x2, s2)

        latent_logits = self.latent_head(h2)
        reward_pred = self.reward_head(h2).squeeze(-1)

        return latent_logits, reward_pred, ((h1, c1), (h2, c2))


# -----------------------------
# Policy Network (Actor-Critic)
# -----------------------------

class PolicyNetwork64(nn.Module):
    """Policy + value network for 8x8 latent codes."""
    def __init__(self, num_actions: int = 5, embedding_dim: int = 32, latent_size: int = 8):
        super().__init__()
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.latent_size = latent_size
        self.embeddings: Optional[nn.Embedding] = None

        self.conv1 = nn.Conv2d(embedding_dim, 256, kernel_size=3, padding=1)
        self.ln1 = ChannelLayerNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.ln2 = ChannelLayerNorm2d(256)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * latent_size * latent_size, 1024),
            nn.LeakyReLU(0.2)
        )
        self.action_head = nn.Linear(1024, num_actions)
        self.value_head = nn.Linear(1024, 1)

    def set_embeddings(self, embeddings: nn.Embedding):
        self.embeddings = embeddings

    def _embed_indices(self, idx: torch.Tensor) -> torch.Tensor:
        if self.embeddings is None:
            raise RuntimeError("Embeddings not set.")
        z = self.embeddings(idx)
        return z.permute(0, 3, 1, 2).contiguous()

    def forward(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._embed_indices(indices)
        x = F.leaky_relu(self.ln1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.ln2(self.conv2(x)), 0.2)
        feat = self.fc(x)
        logits = self.action_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value


# -----------------------------
# Latent-space Simulator
# -----------------------------

class LatentSpaceEnv:
    """Simulated environment using the learned dynamics model."""
    def __init__(self, dynamics: DynamicsModel64, horizon: int = 50, device: str = "cuda"):
        self.dynamics = dynamics
        self.horizon = horizon
        self.device = torch.device(device)
        self.t = 0
        self.indices: Optional[torch.Tensor] = None
        self.state = None

    def reset(self, start_indices: torch.Tensor):
        self.t = 0
        self.indices = start_indices.to(self.device)
        self.state = None
        return self.indices

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict]:
        assert self.indices is not None
        latent_logits, reward_pred, self.state = self.dynamics(self.indices, action, self.state)

        probs = torch.softmax(latent_logits, dim=1)
        b, k, h, w = probs.shape
        probs_flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, k)
        next_idx = torch.multinomial(probs_flat, 1).view(b, h, w)

        self.indices = next_idx
        self.t += 1
        done = self.t >= self.horizon
        return self.indices, reward_pred, done, {}


# -----------------------------
# PPO Config
# -----------------------------

@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 64


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T, B = rewards.shape
    adv = torch.zeros(T, B, device=rewards.device)
    last_gae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * values[t + 1] * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values[:-1]
    return adv, returns


# -----------------------------
# VQ-VAE Agent (compatible with dreamer.py interface)
# -----------------------------

class VQVAEAgent(nn.Module):
    """
    VQ-VAE based world model agent, compatible with the Dreamer training loop.
    """
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super().__init__()
        self._config = config
        self._logger = logger
        self._dataset = dataset
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._metrics = {}

        # Model dimensions
        self.latent_size = 8  # For 64x64 images
        self.embedding_dim = getattr(config, 'vqvae_embedding_dim', 32)
        self.num_embeddings = getattr(config, 'vqvae_num_embeddings', 128)
        self.hidden_channels = getattr(config, 'vqvae_hidden_channels', 256)
        
        # Get image channels
        if 'image' in obs_space.spaces:
            img_shape = obs_space.spaces['image'].shape
            self.in_channels = img_shape[-1] if len(img_shape) == 3 else img_shape[0]
        else:
            self.in_channels = 3
        
        # Get action count
        self.num_actions = act_space.n if hasattr(act_space, 'n') else int(np.prod(act_space.shape))

        # Build models
        self.vqvae = VQVAE64(
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            num_embeddings=self.num_embeddings,
        )
        self.dynamics = DynamicsModel64(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            num_actions=self.num_actions,
            hidden_channels=self.hidden_channels,
            latent_size=self.latent_size,
        )
        self.policy = PolicyNetwork64(
            num_actions=self.num_actions,
            embedding_dim=self.embedding_dim,
            latent_size=self.latent_size,
        )

        # Share embeddings
        self.dynamics.set_embeddings(self.vqvae.vq.embeddings)
        self.policy.set_embeddings(self.vqvae.vq.embeddings)

        # Optimizers
        self.vqvae_opt = torch.optim.Adam(self.vqvae.parameters(), lr=config.model_lr)
        
        dyn_base, dyn_reward = [], []
        for n, p in self.dynamics.named_parameters():
            if p.requires_grad:
                if n.startswith("reward_head"):
                    dyn_reward.append(p)
                else:
                    dyn_base.append(p)
        self.dynamics_opt = torch.optim.Adam([
            {"params": dyn_base, "lr": config.model_lr},
            {"params": dyn_reward, "lr": config.model_lr * 3},
        ])
        self.policy_opt = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.actor["lr"],
            eps=config.actor["eps"],
        )

        self.ppo_cfg = PPOConfig()
        self._should_train = self._make_should_train(config)
        self._should_pretrain = self._make_should_pretrain(config)
        self._warmup_done = False
        
        # Compatibility with Dreamer interface
        self._wm = None  # VQ-VAE doesn't have video_pred like Dreamer

    def _make_should_train(self, config):
        import tools
        batch_steps = config.batch_size * config.batch_length
        return tools.Every(batch_steps / config.train_ratio)

    def _make_should_pretrain(self, config):
        import tools
        return tools.Once()

    def __call__(self, obs, reset, state=None, training=True):
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(self._step)
            )
            for _ in range(steps):
                if self._dataset is not None:
                    self._train(next(self._dataset))
                    self._update_count += 1
                    self._metrics["update_count"] = self._update_count
            
            # Log metrics periodically
            if self._step > 0 and self._step % (self._config.log_every // self._config.action_repeat) == 0:
                for name, values in self._metrics.items():
                    if isinstance(values, list) and len(values) > 0:
                        self._logger.scalar(name, float(np.mean(values)))
                        self._metrics[name] = []

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        with torch.no_grad():
            # Preprocess observation
            if 'image' in obs:
                img = obs['image']
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                img = img.float()
                if img.max() > 1.5:
                    img = img / 255.0
                # Ensure NCHW format
                if img.dim() == 4 and img.shape[-1] in (1, 3, 4):
                    img = img.permute(0, 3, 1, 2)
                img = img.to(self._config.device)
                
                indices = self.vqvae.encode_indices(img)
                logits, value = self.policy(indices)
                
                if training:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                else:
                    action = torch.argmax(logits, dim=-1)
                
                # Convert to one-hot
                action_onehot = F.one_hot(action, num_classes=self.num_actions).float()
                logprob = torch.distributions.Categorical(logits=logits).log_prob(action)
                
                policy_output = {"action": action_onehot, "logprob": logprob}
                return policy_output, (indices, action)
            else:
                # Fallback for non-image observations
                batch_size = 1
                action = torch.randint(0, self.num_actions, (batch_size,), device=self._config.device)
                action_onehot = F.one_hot(action, num_classes=self.num_actions).float()
                logprob = torch.zeros(batch_size, device=self._config.device)
                return {"action": action_onehot, "logprob": logprob}, None

    def _train(self, data):
        metrics = {}

        # Preprocess data
        if 'image' in data:
            images = data['image']
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images)
            images = images.float()
            if images.max() > 1.5:
                images = images / 255.0
            # BTHWC -> BTCHW
            if images.dim() == 5 and images.shape[-1] in (1, 3, 4):
                images = images.permute(0, 1, 4, 2, 3)
            images = images.to(self._config.device)
            
            actions = data['action']
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions)
            actions = actions.to(self._config.device)
            # Convert one-hot to indices if needed
            if actions.dim() == 3:
                actions = actions.argmax(dim=-1)
            
            rewards = data['reward']
            if isinstance(rewards, np.ndarray):
                rewards = torch.from_numpy(rewards)
            rewards = rewards.float().to(self._config.device)

            # Train VQ-VAE
            vq_metrics = self._train_vqvae(images)
            metrics.update(vq_metrics)

            # Train dynamics
            dyn_metrics = self._train_dynamics(images, actions, rewards)
            metrics.update(dyn_metrics)

            # Train policy in imagination
            ppo_metrics = self._train_policy_imagination(images)
            metrics.update(ppo_metrics)

        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _train_vqvae(self, images):
        """Train VQ-VAE on batch of images."""
        self.vqvae.train()
        B, T, C, H, W = images.shape
        images_flat = images.view(B * T, C, H, W)

        out = self.vqvae(images_flat)
        recon_nll = continuous_bernoulli_nll(images_flat, out["logits"])
        vq_loss = out["vq_loss"]
        loss = recon_nll + vq_loss

        self.vqvae_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vqvae.parameters(), 1.0)
        self.vqvae_opt.step()

        return {
            "vqvae_loss": loss.item(),
            "vq_loss": vq_loss.item(),
            "recon_nll": recon_nll.item(),
        }

    def _train_dynamics(self, images, actions, rewards):
        """Train dynamics model on sequences."""
        self.dynamics.train()
        self.vqvae.eval()

        B, T, C, H, W = images.shape
        images_flat = images.view(B * T, C, H, W)
        
        with torch.no_grad():
            indices_flat = self.vqvae.encode_indices(images_flat)
        indices = indices_flat.view(B, T, self.latent_size, self.latent_size)

        state = None
        latent_loss = 0.0
        reward_loss = 0.0
        seq_len = T - 1

        for t in range(seq_len):
            idx_t = indices[:, t]
            idx_tp1 = indices[:, t + 1]
            a_t = actions[:, t]
            r_t = rewards[:, t]

            latent_logits, reward_pred, state = self.dynamics(idx_t, a_t, state)
            latent_loss = latent_loss + F.cross_entropy(latent_logits, idx_tp1.long())
            reward_loss = reward_loss + F.mse_loss(reward_pred, r_t)

        latent_loss = latent_loss / seq_len
        reward_loss = reward_loss / seq_len
        loss = latent_loss + 0.1 * reward_loss

        self.dynamics_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
        self.dynamics_opt.step()

        return {
            "dynamics_loss": loss.item(),
            "latent_ce": latent_loss.item(),
            "reward_mse": reward_loss.item(),
        }

    def _train_policy_imagination(self, images):
        """Train policy using imagined rollouts."""
        self.policy.train()
        self.dynamics.eval()

        B, T, C, H, W = images.shape
        horizon = min(self._config.imag_horizon, 15)

        # Sample starting states
        t_starts = torch.randint(0, T, (B,))
        start_images = torch.stack([images[b, t_starts[b]] for b in range(B)])
        
        with torch.no_grad():
            start_indices = self.vqvae.encode_indices(start_images)

        sim_env = LatentSpaceEnv(self.dynamics, horizon=horizon, device=str(self._config.device))
        
        obs_idx_list, act_list, logp_list, rew_list, done_list, val_list = [], [], [], [], [], []
        
        sim_env.reset(start_indices)
        
        for t in range(horizon):
            logits, value = self.policy(sim_env.indices)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

            next_idx, reward, done, _ = sim_env.step(action)

            obs_idx_list.append(sim_env.indices.detach())
            act_list.append(action.detach())
            logp_list.append(logp.detach())
            rew_list.append(reward.detach())
            done_list.append(torch.full((B,), float(done), device=self._config.device))
            val_list.append(value.detach())

        with torch.no_grad():
            _, last_value = self.policy(sim_env.indices)
        val_list.append(last_value.detach())

        obs_idx = torch.stack(obs_idx_list, dim=0)
        actions = torch.stack(act_list, dim=0)
        old_logp = torch.stack(logp_list, dim=0)
        rewards = torch.stack(rew_list, dim=0)
        dones = torch.stack(done_list, dim=0).bool()
        values = torch.stack(val_list, dim=0)

        adv, returns = compute_gae(rewards, values, dones, 
                                    gamma=self._config.discount, 
                                    lam=self._config.discount_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        Th, Bh = rewards.shape
        flat_n = Th * Bh

        obs_flat = obs_idx.view(flat_n, self.latent_size, self.latent_size)
        act_flat = actions.view(flat_n)
        old_logp_flat = old_logp.view(flat_n)
        adv_flat = adv.view(flat_n)
        ret_flat = returns.view(flat_n)

        policy_losses, value_losses, entropies = [], [], []

        for _ in range(self.ppo_cfg.ppo_epochs):
            perm = torch.randperm(flat_n, device=self._config.device)
            for start in range(0, flat_n, self.ppo_cfg.minibatch_size):
                mb_idx = perm[start:start + self.ppo_cfg.minibatch_size]
                mb_obs = obs_flat[mb_idx]
                mb_act = act_flat[mb_idx]
                mb_old_logp = old_logp_flat[mb_idx]
                mb_adv = adv_flat[mb_idx]
                mb_ret = ret_flat[mb_idx]

                logits, value = self.policy(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_cfg.clip_range, 1.0 + self.ppo_cfg.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, mb_ret)

                loss = policy_loss + self.ppo_cfg.value_coef * value_loss - self.ppo_cfg.entropy_coef * entropy

                self.policy_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.ppo_cfg.max_grad_norm)
                self.policy_opt.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        return {
            "ppo_policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "ppo_value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "ppo_entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def video_pred(self, data):
        """Generate video prediction for logging (similar to Dreamer)."""
        return None  # VQ-VAE doesn't do video prediction the same way
