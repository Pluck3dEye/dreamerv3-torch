"""
Self-contained implementation of the "Smaller World Models for RL" recipe (Robine et al., arXiv:2010.05767v2)
adapted to a generic Gym environment.

Key fixes vs the previous draft:
- Adds an in-file VQ-VAE (96x96x4 -> 6x6 discrete codes, codebook size 128, embedding dim 32).
- Uses a Continuous Bernoulli decoder likelihood (with stable log-normalizer) instead of MSE.
- Warmup: 50 epochs with a higher LR for VQ-VAE, then lower LR.
- Slower representation drift: VQ-VAE optimizer steps every N batches (default N=2) after warmup.
- Dynamics reward head uses a higher LR and reward loss is downweighted.
- Policy input uses the same 32-channel embeddings as the dynamics model for fidelity.

NOTE:
- This file is written to be readable and modifiable. Exact hyperparameters may still need tuning for your target env.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collections import deque


# -----------------------------
# Utilities
# -----------------------------

def to_torch(x, device: torch.device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


class ChannelLayerNorm2d(nn.Module):
    """
    LayerNorm over channel dimension for 2D feature maps, keeping H,W intact.
    This behaves like LayerNorm(C) applied per spatial location.
    """
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H, W, C) -> LN -> back
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


def continuous_bernoulli_nll(x: torch.Tensor, logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Negative log-likelihood for Continuous Bernoulli distribution.

    x in [0,1], logits are unconstrained.
    From "The Continuous Bernoulli: Fixing a Pervasive Error in Variational Autoencoders".

    log p(x|λ) = x*log λ + (1-x)*log(1-λ) + log C(λ)
    where C(λ) = (2*atanh(1-2λ)) / (1-2λ)
    and λ = sigmoid(logits).

    Returns mean NLL over batch.
    """
    x = torch.clamp(x, 0.0, 1.0)
    lam = torch.sigmoid(logits)
    lam = torch.clamp(lam, eps, 1.0 - eps)

    # BCE term (negative of x log λ + (1-x) log(1-λ))
    bce = F.binary_cross_entropy(lam, x, reduction='none')

    # log normalizer log C(λ)
    t = 1.0 - 2.0 * lam  # in (-1,1)
    abs_t = torch.abs(t)

    # Stable computation of logC:
    # logC = log(2*atanh(t)/t). Handle near t=0 with series expansion:
    # logC ≈ log(2) + t^2/3 + 2 t^4/15  (good near 0)
    small = abs_t < 1e-3
    # series
    t2 = t * t
    logC_series = math.log(2.0) + (t2 / 3.0) + (2.0 * t2 * t2 / 15.0)

    # general
    atanh_t = 0.5 * (torch.log1p(t) - torch.log1p(-t))  # atanh(t)
    logC_general = torch.log(2.0 * atanh_t / t)

    logC = torch.where(small, logC_series, logC_general)

    # NLL = BCE - logC
    nll = bce - logC
    return nll.mean()


# -----------------------------
# VQ-VAE (96x96x4 -> 6x6 codes)
# -----------------------------

class Encoder(nn.Module):
    """
    Encoder producing 6x6xD latents from 96x96x4 observations.
    Uses stride-2 convs: 96->48->24->12->6.
    """
    def __init__(self, in_channels: int = 4, hidden: int = 128, embedding_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 4, stride=2, padding=1),  # 96->48
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1),       # 48->24
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1),       # 24->12
            nn.ReLU(),
            nn.Conv2d(hidden, embedding_dim, 4, stride=2, padding=1) # 12->6
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    Decoder from 6x6xD back to 96x96x4 logits (for continuous Bernoulli likelihood).
    """
    def __init__(self, embedding_dim: int = 32, hidden: int = 128, out_channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden, 4, stride=2, padding=1),  # 6->12
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1),         # 12->24
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1),         # 24->48
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, out_channels, 4, stride=2, padding=1)    # 48->96 (logits)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VectorQuantizer(nn.Module):
    """
    Standard VQ layer with straight-through estimator.
    """
    def __init__(self, num_embeddings: int = 128, embedding_dim: int = 32, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_e: (B, D, H, W)
        Returns:
          z_q: (B, D, H, W) quantized
          vq_loss: scalar
          indices: (B, H, W) long indices into codebook
        """
        b, d, h, w = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous().view(-1, d)  # (BHW, D)

        # distances to embeddings
        emb = self.embeddings.weight  # (K, D)
        dist = (
            (z ** 2).sum(dim=1, keepdim=True)
            - 2.0 * z @ emb.t()
            + (emb ** 2).sum(dim=1, keepdim=True).t()
        )  # (BHW, K)

        indices = torch.argmin(dist, dim=1)  # (BHW,)
        z_q = self.embeddings(indices).view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        # losses
        # codebook loss: move embeddings toward encoder outputs
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        # commitment loss: move encoder outputs toward embeddings
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        # straight-through
        z_q_st = z_e + (z_q - z_e).detach()
        indices = indices.view(b, h, w)
        return z_q_st, vq_loss, indices


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        embedding_dim: int = 32,
        num_embeddings: int = 128,
        commitment_cost: float = 0.25,
        hidden: int = 128,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, hidden=hidden, embedding_dim=embedding_dim)
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder(embedding_dim=embedding_dim, hidden=hidden, out_channels=in_channels)

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x)
        _, _, indices = self.vq(z_e)
        return indices  # (B,6,6)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        logits = self.decoder(z_q)  # logits for continuous bernoulli
        return {"logits": logits, "vq_loss": vq_loss, "indices": indices, "z_q": z_q}


# -----------------------------
# ConvLSTM Dynamics Model
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


class DynamicsModel(nn.Module):
    """
    Predicts next code indices distribution and reward category from current code embeddings + action one-hot.
    - embedding channels: 32
    - action channels: 16
    Input: (B, 6, 6) indices + action int -> embeddings -> (B, 32, 6, 6) + action one-hot -> (B, 48, 6, 6)
    """
    def __init__(self, num_embeddings: int = 128, embedding_dim: int = 32, num_actions: int = 16, hidden_channels: int = 256):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        self.hidden_channels = hidden_channels

        self.embeddings: Optional[nn.Embedding] = None

        self.convlstm1 = ConvLSTMCell(embedding_dim + num_actions, hidden_channels)
        self.convlstm2 = ConvLSTMCell(hidden_channels + num_actions, hidden_channels)

        # Next-latent head: outputs (B, num_embeddings, 6, 6)
        self.latent_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 256, kernel_size=3, padding=1),
            ChannelLayerNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, num_embeddings, kernel_size=1)
        )

        # Reward head: outputs logits over 3 categories {-1,0,1}
        self.reward_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
            ChannelLayerNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 3)
        )

    def set_embeddings(self, embeddings: nn.Embedding):
        self.embeddings = embeddings

    def _embed_indices(self, idx: torch.Tensor) -> torch.Tensor:
        if self.embeddings is None:
            raise RuntimeError("Embeddings not set. Call set_embeddings(vqvae.vq.embeddings).")
        # idx: (B,6,6) -> (B,6,6, D) -> (B,D,6,6)
        z = self.embeddings(idx)  # (B,6,6,D)
        return z.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        indices: torch.Tensor,  # (B,6,6)
        actions: torch.Tensor,  # (B,) int
        state: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        device = indices.device
        b = indices.shape[0]

        z = self._embed_indices(indices)  # (B, D, 6, 6)

        # One-hot action planes
        a_onehot = F.one_hot(actions.long(), num_classes=self.num_actions).float().to(device)  # (B,A)
        a_planes = a_onehot.view(b, self.num_actions, 1, 1).expand(b, self.num_actions, 6, 6)

        x1 = torch.cat([z, a_planes], dim=1)  # (B, D+A, 6, 6)

        if state is None:
            s1 = self.convlstm1.init_state(b, (6, 6), device)
            s2 = self.convlstm2.init_state(b, (6, 6), device)
        else:
            s1, s2 = state

        h1, c1 = self.convlstm1(x1, s1)
        x2 = torch.cat([h1, a_planes], dim=1)
        h2, c2 = self.convlstm2(x2, s2)

        latent_logits = self.latent_head(h2)   # (B, K, 6, 6)
        reward_logits = self.reward_head(h2)   # (B, 3)

        return latent_logits, reward_logits, ((h1, c1), (h2, c2))


# -----------------------------
# Policy network (PPO) in latent space
# -----------------------------

class PolicyNetwork(nn.Module):
    """
    Policy + value network operating on latent embeddings (B,32,6,6).
    Architecture per paper: 2 conv layers (3x3, 256ch) w/ LN + LeakyReLU, then FC 1024.
    Outputs action logits and value.
    """
    def __init__(self, num_actions: int = 16, embedding_dim: int = 32):
        super().__init__()
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.embeddings: Optional[nn.Embedding] = None

        self.conv1 = nn.Conv2d(embedding_dim, 256, kernel_size=3, padding=1)
        self.ln1 = ChannelLayerNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.ln2 = ChannelLayerNorm2d(256)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.LeakyReLU(0.2)
        )
        self.action_head = nn.Linear(1024, num_actions)
        self.value_head = nn.Linear(1024, 1)

    def set_embeddings(self, embeddings: nn.Embedding):
        self.embeddings = embeddings

    def _embed_indices(self, idx: torch.Tensor) -> torch.Tensor:
        if self.embeddings is None:
            raise RuntimeError("Embeddings not set. Call set_embeddings(vqvae.vq.embeddings).")
        z = self.embeddings(idx)  # (B,6,6,D)
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
# Experience buffer for real trajectories
# -----------------------------

class ExperienceBuffer:
    def __init__(self, maxlen: int = 200_000):
        self.obs: Deque[np.ndarray] = deque(maxlen=maxlen)
        self.actions: Deque[int] = deque(maxlen=maxlen)
        self.rewards: Deque[int] = deque(maxlen=maxlen)

    def add(self, obs: np.ndarray, action: int, reward: int):
        self.obs.append(obs)
        self.actions.append(int(action))
        self.rewards.append(int(reward))

    def __len__(self):
        return len(self.obs)

    def sample_sequences(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Samples sequences of length seq_len with aligned (obs_t, action_t, reward_t, obs_{t+1}).
        Returns tensors:
          obs: (B, seq_len+1, C, H, W)
          actions: (B, seq_len)
          rewards: (B, seq_len)
        """
        assert len(self.obs) > seq_len + 1, "Not enough data in buffer."

        max_start = len(self.obs) - (seq_len + 1)
        starts = np.random.randint(0, max_start, size=batch_size)

        obs_seq = []
        act_seq = []
        rew_seq = []
        for s in starts:
            o = [self.obs[s + t] for t in range(seq_len + 1)]
            a = [self.actions[s + t] for t in range(seq_len)]
            r = [self.rewards[s + t] for t in range(seq_len)]
            obs_seq.append(np.stack(o, axis=0))
            act_seq.append(np.array(a, dtype=np.int64))
            rew_seq.append(np.array(r, dtype=np.int64))

        obs_seq = np.stack(obs_seq, axis=0)  # (B, T+1, H, W, C) or (B,T+1,C,H,W) depending on env
        # Normalize/transpose later in trainer.
        return {
            "obs": torch.from_numpy(obs_seq),
            "actions": torch.from_numpy(np.stack(act_seq, axis=0)),
            "rewards": torch.from_numpy(np.stack(rew_seq, axis=0)),
        }


# -----------------------------
# Latent-space environment (simulator)
# -----------------------------

class LatentSpaceEnv:
    """
    Simulated environment using the learned dynamics model.
    The "state" is the current 6x6 code indices, plus ConvLSTM hidden state.
    Episodes are fixed-length (horizon) like in the paper to avoid terminal prediction.
    """
    def __init__(self, dynamics: DynamicsModel, horizon: int = 50, device: str = "cuda"):
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
        latent_logits, reward_logits, self.state = self.dynamics(self.indices, action, self.state)

        # sample next indices per spatial location
        # latent_logits: (B,K,6,6)
        probs = torch.softmax(latent_logits, dim=1)
        b, k, h, w = probs.shape
        probs_flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, k)
        next_idx = torch.multinomial(probs_flat, 1).view(b, h, w)

        # reward category sample -> map {0,1,2} to {-1,0,1}
        r_cat = torch.distributions.Categorical(logits=reward_logits).sample()
        reward = (r_cat - 1).float()

        self.indices = next_idx
        self.t += 1
        done = self.t >= self.horizon
        return self.indices, reward, done, {}


# -----------------------------
# PPO helper
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
    """
    rewards: (T,B)
    values: (T+1,B)
    dones: (T,B) boolean
    """
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
# Trainer
# -----------------------------

class WorldModelTrainer:
    def __init__(
        self,
        env_name: str,
        device: str = "cuda",
        num_actions: int = 16,
        obs_channels: int = 4,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.env = gym.make(env_name)

        # Models
        self.vqvae = VQVAE(in_channels=obs_channels, embedding_dim=32, num_embeddings=128).to(self.device)
        self.dynamics = DynamicsModel(num_embeddings=128, embedding_dim=32, num_actions=num_actions).to(self.device)
        self.policy = PolicyNetwork(num_actions=num_actions, embedding_dim=32).to(self.device)

        # Share embeddings
        self.dynamics.set_embeddings(self.vqvae.vq.embeddings)
        self.policy.set_embeddings(self.vqvae.vq.embeddings)

        # Buffers
        self.real_buffer = ExperienceBuffer()

        # Optimizers
        self.vqvae_optimizer = torch.optim.Adam(self.vqvae.parameters(), lr=1e-3)  # warmup lr
        # dynamics: param groups with higher LR for reward head
        dyn_base = []
        dyn_reward = []
        for n, p in self.dynamics.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("reward_head"):
                dyn_reward.append(p)
            else:
                dyn_base.append(p)
        self.dynamics_optimizer = torch.optim.Adam(
            [{"params": dyn_base, "lr": 3e-4},
             {"params": dyn_reward, "lr": 1e-3}],
        )
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=2.5e-4, eps=1e-5)

        self.ppo_cfg = PPOConfig()

    # ---- gym API compatibility ----
    def _env_reset(self):
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
            return obs
        return out

    def _env_step(self, action):
        out = self.env.step(action)
        # gymnasium / newer gym: (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, reward, done, info
        # older gym: (obs, reward, done, info)
        obs, reward, done, info = out
        return obs, reward, bool(done), info

    # ---- observation handling ----
    def _prep_obs(self, obs: np.ndarray) -> torch.Tensor:
        """
        Converts env observation to (C,H,W) float in [0,1].
        Expected obs to be HxWxC or CxHxW.
        """
        x = obs
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
            # likely C,H,W already
            pass
        elif x.ndim == 3:
            # H,W,C -> C,H,W
            x = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported obs shape: {x.shape}")

        x = x.astype(np.float32)
        # heuristic normalization if pixels in [0,255]
        if x.max() > 1.5:
            x = x / 255.0
        x = np.clip(x, 0.0, 1.0)
        return torch.from_numpy(x)

    @torch.no_grad()
    def encode_obs(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """
        obs_batch: (B,C,H,W) float in [0,1]
        returns indices: (B,6,6)
        """
        obs_batch = obs_batch.to(self.device)
        return self.vqvae.encode_indices(obs_batch)

    # ---- data collection ----
    def collect_real_experience(self, num_steps: int, random_policy: bool = True):
        obs = self._env_reset()
        for _ in range(num_steps):
            if random_policy or len(self.real_buffer) < 1000:
                action = self.env.action_space.sample()
            else:
                # policy in latent space
                obs_t = self._prep_obs(obs).unsqueeze(0).to(self.device)
                idx = self.encode_obs(obs_t)
                logits, _ = self.policy(idx)
                action = torch.distributions.Categorical(logits=logits).sample().item()

            next_obs, reward, done, _info = self._env_step(action)

            # reward category expected {-1,0,1} in paper; clamp/map if needed
            # Here we map any reward to {-1,0,1} by sign, preserving 0.
            r_cat = int(np.sign(reward))
            self.real_buffer.add(self._prep_obs(obs).numpy(), action, r_cat)

            obs = next_obs
            if done:
                obs = self._env_reset()

    # ---- VQ-VAE training ----
    def train_vqvae(
        self,
        batch_size: int = 64,
        steps: int = 500,
        update_every: int = 1,
    ) -> Dict[str, float]:
        """
        Trains VQ-VAE for a number of gradient steps sampled from the real buffer.
        update_every: perform optimizer step only every N batches (N=2 recommended after warmup).
        """
        assert len(self.real_buffer) > 1000, "Need more real experience before training."
        self.vqvae.train()
        losses = []
        vq_losses = []
        recon_losses = []

        step_counter = 0
        for _ in range(steps):
            # sample random obs frames
            idxs = np.random.randint(0, len(self.real_buffer.obs), size=batch_size)
            obs = np.stack([self.real_buffer.obs[i] for i in idxs], axis=0)  # (B,C,H,W)
            obs_t = to_torch(obs, self.device)

            out = self.vqvae(obs_t)
            logits = out["logits"]
            vq_loss = out["vq_loss"]
            recon_nll = continuous_bernoulli_nll(obs_t, logits)

            loss = recon_nll + vq_loss

            loss.backward()

            step_counter += 1
            if step_counter % update_every == 0:
                torch.nn.utils.clip_grad_norm_(self.vqvae.parameters(), 1.0)
                self.vqvae_optimizer.step()
                self.vqvae_optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            vq_losses.append(vq_loss.item())
            recon_losses.append(recon_nll.item())

        return {
            "vqvae_loss": float(np.mean(losses)),
            "vq_loss": float(np.mean(vq_losses)),
            "recon_nll": float(np.mean(recon_losses)),
        }

    # ---- Dynamics training ----
    def train_dynamics(
        self,
        batch_size: int = 64,
        seq_len: int = 16,
        steps: int = 500,
        reward_loss_scale: float = 0.1,
    ) -> Dict[str, float]:
        """
        Trains dynamics on sequences from real buffer.
        Predict next indices and reward category per time step.
        """
        assert len(self.real_buffer) > 5000, "Need more real experience before training dynamics."
        self.dynamics.train()
        self.vqvae.eval()  # freeze encoder target during dynamics update step

        latent_losses = []
        reward_losses = []
        total_losses = []

        for _ in range(steps):
            batch = self.real_buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len)
            obs_seq = batch["obs"]  # (B, T+1, C, H, W)
            actions = batch["actions"]  # (B, T)
            rewards = batch["rewards"]  # (B, T)

            obs_seq = obs_seq.to(self.device, dtype=torch.float32)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)

            # encode all obs frames to indices
            b, t1, c, h, w = obs_seq.shape
            obs_flat = obs_seq.view(b * t1, c, h, w)
            with torch.no_grad():
                idx_flat = self.encode_obs(obs_flat)  # (B*(T+1),6,6)
            idx_seq = idx_flat.view(b, t1, 6, 6)

            # rollout ConvLSTM over time
            state = None
            latent_loss = 0.0
            reward_loss = 0.0
            for t in range(seq_len):
                idx_t = idx_seq[:, t]           # (B,6,6)
                idx_tp1 = idx_seq[:, t + 1]     # (B,6,6)
                a_t = actions[:, t]
                r_t = rewards[:, t]             # in {-1,0,1}
                # map to 0,1,2
                r_cls = (r_t + 1).long().clamp(0, 2)

                latent_logits, reward_logits, state = self.dynamics(idx_t, a_t, state)

                # latent CE over each spatial position
                # latent_logits: (B,K,6,6), target: (B,6,6)
                latent_loss = latent_loss + F.cross_entropy(latent_logits, idx_tp1.long())

                reward_loss = reward_loss + F.cross_entropy(reward_logits, r_cls)

            latent_loss = latent_loss / seq_len
            reward_loss = reward_loss / seq_len

            loss = latent_loss + reward_loss_scale * reward_loss

            self.dynamics_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
            self.dynamics_optimizer.step()

            latent_losses.append(latent_loss.item())
            reward_losses.append(reward_loss.item())
            total_losses.append(loss.item())

        return {
            "dynamics_loss": float(np.mean(total_losses)),
            "latent_ce": float(np.mean(latent_losses)),
            "reward_ce": float(np.mean(reward_losses)),
        }

    # ---- PPO training in world model ----
    @torch.no_grad()
    def _sample_start_indices(self, batch_size: int) -> torch.Tensor:
        idxs = np.random.randint(0, len(self.real_buffer.obs), size=batch_size)
        obs = np.stack([self.real_buffer.obs[i] for i in idxs], axis=0)  # (B,C,H,W)
        obs_t = to_torch(obs, self.device)
        return self.encode_obs(obs_t)

    def train_policy_in_world_model(
        self,
        simulated_episodes: int = 64,
        horizon: int = 50,
    ) -> Dict[str, float]:
        """
        Collects rollouts in the latent-space env and runs PPO updates.
        """
        self.policy.train()
        self.dynamics.eval()

        sim_env = LatentSpaceEnv(self.dynamics, horizon=horizon, device=str(self.device))

        # storage
        obs_idx_list = []
        act_list = []
        logp_list = []
        rew_list = []
        done_list = []
        val_list = []

        batch_size = simulated_episodes
        indices = self._sample_start_indices(batch_size)
        sim_env.reset(indices)

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
            done_list.append(torch.full((batch_size,), float(done), device=self.device))
            val_list.append(value.detach())

        # bootstrap value for last step
        with torch.no_grad():
            _, last_value = self.policy(sim_env.indices)
        val_list.append(last_value.detach())

        # stack
        obs_idx = torch.stack(obs_idx_list, dim=0)  # (T,B,6,6)
        actions = torch.stack(act_list, dim=0)      # (T,B)
        old_logp = torch.stack(logp_list, dim=0)    # (T,B)
        rewards = torch.stack(rew_list, dim=0)      # (T,B)
        dones = torch.stack(done_list, dim=0).bool()# (T,B)
        values = torch.stack(val_list, dim=0)       # (T+1,B)

        adv, returns = compute_gae(rewards, values, dones, gamma=self.ppo_cfg.gamma, lam=self.ppo_cfg.gae_lambda)
        # normalize advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T, B = rewards.shape
        flat_n = T * B

        # flatten for PPO minibatches
        obs_flat = obs_idx.view(flat_n, 6, 6)
        act_flat = actions.view(flat_n)
        old_logp_flat = old_logp.view(flat_n)
        adv_flat = adv.view(flat_n)
        ret_flat = returns.view(flat_n)

        # PPO updates
        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(self.ppo_cfg.ppo_epochs):
            perm = torch.randperm(flat_n, device=self.device)
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

                self.policy_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.ppo_cfg.max_grad_norm)
                self.policy_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        return {
            "ppo_policy_loss": float(np.mean(policy_losses)),
            "ppo_value_loss": float(np.mean(value_losses)),
            "ppo_entropy": float(np.mean(entropies)),
        }

    # ---- Evaluation on real env ----
    @torch.no_grad()
    def evaluate_policy(self, episodes: int = 5) -> Dict[str, float]:
        self.policy.eval()
        returns = []
        for _ in range(episodes):
            obs = self._env_reset()
            done = False
            ep_ret = 0.0
            while not done:
                obs_t = self._prep_obs(obs).unsqueeze(0).to(self.device)
                idx = self.encode_obs(obs_t)
                logits, _ = self.policy(idx)
                action = torch.distributions.Categorical(logits=logits).sample().item()
                obs, reward, done, _ = self._env_step(action)
                ep_ret += float(reward)
            returns.append(ep_ret)
        return {"eval_return_mean": float(np.mean(returns)), "eval_return_std": float(np.std(returns))}

    # ---- Full iterative training loop ----
    def full_training(
        self,
        iterations: int = 15,
        initial_real_steps: int = 12_800,
        subsequent_real_steps: int = 6_400,
        vqvae_warmup_steps: int = 2000,
        vqvae_steps_per_iter: int = 500,
        dynamics_steps_per_iter: int = 500,
        ppo_sim_episodes: int = 64,
        horizon: int = 50,
    ):
        logs = []

        for it in range(iterations):
            # 1) collect real data
            n_real = initial_real_steps if it == 0 else subsequent_real_steps
            self.collect_real_experience(n_real, random_policy=(it == 0))
            print(f"[Iter {it}] Collected {n_real} real steps. Buffer size={len(self.real_buffer)}")

            # 2) train VQ-VAE
            if it == 0:
                # warmup with higher LR (already set to 1e-3)
                vq_log = self.train_vqvae(steps=vqvae_warmup_steps, update_every=1)
                # after warmup, drop LR to 3e-4
                for pg in self.vqvae_optimizer.param_groups:
                    pg["lr"] = 3e-4
                print(f"[Iter {it}] VQ-VAE warmup:", vq_log)
            else:
                # slower updates: step every 2 batches to reduce target drift
                vq_log = self.train_vqvae(steps=vqvae_steps_per_iter, update_every=2)
                print(f"[Iter {it}] VQ-VAE:", vq_log)

            # 3) train dynamics
            dyn_log = self.train_dynamics(steps=dynamics_steps_per_iter)
            print(f"[Iter {it}] Dynamics:", dyn_log)

            # 4) train policy in world model
            ppo_log = self.train_policy_in_world_model(simulated_episodes=ppo_sim_episodes, horizon=horizon)
            print(f"[Iter {it}] PPO:", ppo_log)

            # 5) evaluate
            eval_log = self.evaluate_policy(episodes=5)
            print(f"[Iter {it}] Eval:", eval_log)

            row = {"iter": it, **vq_log, **dyn_log, **ppo_log, **eval_log}
            logs.append(row)

        return logs


# -----------------------------
# Example entrypoint
# -----------------------------

if __name__ == "__main__":
    # Change env_name to your target environment.
    # Make sure observations are image-like (C,H,W) or (H,W,C) and have 4 channels.
    env_name = "CarRacing-v2"  # placeholder; replace with your HighwayEnv wrapper if needed

    trainer = WorldModelTrainer(env_name=env_name, device="cuda", num_actions=16, obs_channels=4)
    trainer.full_training()