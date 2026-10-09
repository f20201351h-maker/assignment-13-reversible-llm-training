"""A tied-weight decoder with ordinary and genuinely reconstructive stacks.

The two custom autograd functions keep only stack boundary tensors. During
backward they invert and differentiate one transformer block at a time.
The embedding, bootstrap, final norm, and output head retain their ordinary
autograd behavior and are included in all reported peak-memory measurements.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 10_000
    block_size: int = 512
    n_layer: int = 9
    n_head: int = 6
    n_embd: int = 384
    ffn_mult: int = 4
    dropout: float = 0.0
    integrator: str = "conventional"
    backward_mode: str = "stored"
    step_size: float = 0.25
    blend: float = 1.0

    def validate(self) -> None:
        if self.n_embd % self.n_head or self.n_layer < 1 or self.block_size < 1:
            raise ValueError("Invalid width, head count, depth, or context length")
        if self.dropout != 0:
            raise ValueError("Reconstruction requires dropout=0 in this study")
        if self.integrator not in {"conventional", "midpoint", "coupled_euler", "blended_midpoint"}:
            raise ValueError(f"Unknown integrator: {self.integrator}")
        if self.backward_mode not in {"stored", "reconstructed", "checkpointed"}:
            raise ValueError(f"Unknown backward mode: {self.backward_mode}")
        if self.integrator == "conventional" and self.backward_mode == "reconstructed":
            raise ValueError("Ordinary residual Euler has no explicit inverse")
        if self.integrator != "conventional" and self.backward_mode == "checkpointed":
            raise ValueError("Checkpoint comparison is reserved for the conventional model")
        if self.step_size <= 0 or (self.integrator == "blended_midpoint" and self.blend == 0):
            raise ValueError("Step size and midpoint inverse coefficient must be nonzero")


class CausalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, d))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.up = nn.Linear(cfg.n_embd, cfg.ffn_mult * cfg.n_embd, bias=False)
        self.down = nn.Linear(cfg.ffn_mult * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(cfg.n_embd, elementwise_affine=True, bias=False)
        self.attn = CausalAttention(cfg)
        self.mlp_norm = nn.LayerNorm(cfg.n_embd, elementwise_affine=True, bias=False)
        self.mlp = FeedForward(cfg)

    def attn_update(self, x: Tensor) -> Tensor:
        return self.attn(self.attn_norm(x))

    def mlp_update(self, x: Tensor) -> Tensor:
        return self.mlp(self.mlp_norm(x))

    def force(self, x: Tensor) -> Tensor:
        a = self.attn_update(x)
        return a + self.mlp_update(x + a)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.force(x)


def _amp_context(enabled: bool, dtype: torch.dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if enabled else nullcontext()


class _MidpointReconstruct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, *args):
        *params, blocks, h, a = args
        ctx.blocks, ctx.h, ctx.a = blocks, h, a
        ctx.amp_enabled = torch.is_autocast_enabled("cuda")
        ctx.amp_dtype = torch.get_autocast_dtype("cuda")
        with torch.no_grad():
            prev = x
            cur = x + 0.5 * h * blocks[0].force(x)
            for block in blocks[1:]:
                prev, cur = cur, a * prev + (1 - a) * cur + 2 * h * block.force(cur)
        ctx.save_for_backward(x, prev, cur, *params)
        return cur

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x0, prev, cur, *params = ctx.saved_tensors
        h, a = ctx.h, ctx.a
        param_indices = {id(p): i for i, p in enumerate(params)}
        param_grads = [torch.zeros_like(p) for p in params]
        g_prev, g_cur = torch.zeros_like(prev), grad_output

        for block in reversed(ctx.blocks[1:]):
            block_params = tuple(block.parameters())
            with torch.no_grad(), _amp_context(ctx.amp_enabled, ctx.amp_dtype):
                older = (cur - (1 - a) * prev - 2 * h * block.force(prev)) / a
            with torch.enable_grad(), _amp_context(ctx.amp_enabled, ctx.amp_dtype):
                source = prev.detach().requires_grad_(True)
                force = block.force(source)
                derivs = torch.autograd.grad(
                    force, (source, *block_params), grad_outputs=2 * h * g_cur
                )
            for p, g in zip(block_params, derivs[1:]):
                param_grads[param_indices[id(p)]] += g
            g_prev, g_cur = a * g_cur, g_prev + (1 - a) * g_cur + derivs[0]
            prev, cur = older, prev

        bootstrap = ctx.blocks[0]
        bootstrap_params = tuple(bootstrap.parameters())
        with torch.enable_grad(), _amp_context(ctx.amp_enabled, ctx.amp_dtype):
            source = x0.detach().requires_grad_(True)
            force = bootstrap.force(source)
            derivs = torch.autograd.grad(
                force, (source, *bootstrap_params), grad_outputs=0.5 * h * g_cur
            )
        for p, g in zip(bootstrap_params, derivs[1:]):
            param_grads[param_indices[id(p)]] += g
        grad_input = g_prev + g_cur + derivs[0]
        return (grad_input, *param_grads, None, None, None)


class MidpointStack(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layer))
        self.h = cfg.step_size
        self.a = cfg.blend if cfg.integrator == "blended_midpoint" else 1.0
        self.reconstructed = cfg.backward_mode == "reconstructed"

    def forward(self, x: Tensor) -> Tensor:
        if self.reconstructed and torch.is_grad_enabled():
            return _MidpointReconstruct.apply(x, *tuple(self.blocks.parameters()), self.blocks, self.h, self.a)
        prev = x
        cur = x + 0.5 * self.h * self.blocks[0].force(x)
        for block in self.blocks[1:]:
            prev, cur = cur, self.a * prev + (1 - self.a) * cur + 2 * self.h * block.force(cur)
        return cur


class _CoupledReconstruct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, *args):
        *params, blocks, h = args
        ctx.blocks, ctx.h = blocks, h
        ctx.amp_enabled = torch.is_autocast_enabled("cuda")
        ctx.amp_dtype = torch.get_autocast_dtype("cuda")
        with torch.no_grad():
            u, v = x, x
            for block in blocks:
                u = u + h * block.attn_update(v)
                v = v + h * block.mlp_update(u)
        ctx.save_for_backward(x, u, v, *params)
        return u, v

    @staticmethod
    def backward(ctx, grad_u: Tensor, grad_v: Tensor):
        x0, u, v, *params = ctx.saved_tensors
        h = ctx.h
        param_indices = {id(p): i for i, p in enumerate(params)}
        param_grads = [torch.zeros_like(p) for p in params]
        for block in reversed(ctx.blocks):
            block_params = tuple(block.parameters())
            with torch.no_grad(), _amp_context(ctx.amp_enabled, ctx.amp_dtype):
                prior_v = v - h * block.mlp_update(u)
                prior_u = u - h * block.attn_update(prior_v)
            with torch.enable_grad(), _amp_context(ctx.amp_enabled, ctx.amp_dtype):
                source_u = prior_u.detach().requires_grad_(True)
                source_v = prior_v.detach().requires_grad_(True)
                next_u = source_u + h * block.attn_update(source_v)
                next_v = source_v + h * block.mlp_update(next_u)
                derivs = torch.autograd.grad(
                    (next_u, next_v), (source_u, source_v, *block_params),
                    grad_outputs=(grad_u, grad_v)
                )
            for p, g in zip(block_params, derivs[2:]):
                param_grads[param_indices[id(p)]] += g
            grad_u, grad_v = derivs[:2]
            u, v = prior_u, prior_v
        return (grad_u + grad_v, *param_grads, None, None)


class CoupledStack(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layer))
        self.h = cfg.step_size
        self.reconstructed = cfg.backward_mode == "reconstructed"

    def forward(self, x: Tensor) -> Tensor:
        if self.reconstructed and torch.is_grad_enabled():
            u, v = _CoupledReconstruct.apply(x, *tuple(self.blocks.parameters()), self.blocks, self.h)
        else:
            u, v = x, x
            for block in self.blocks:
                u = u + self.h * block.attn_update(v)
                v = v + self.h * block.mlp_update(u)
        return 0.5 * (u + v)


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        if cfg.integrator == "conventional":
            self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layer))
            self.stack = None
        elif cfg.integrator in {"midpoint", "blended_midpoint"}:
            self.blocks = None
            self.stack = MidpointStack(cfg)
        else:
            self.blocks = None
            self.stack = CoupledStack(cfg)
        self.final_norm = nn.LayerNorm(cfg.n_embd, elementwise_affine=True, bias=False)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Tensor | None = None, *, return_token_losses: bool = False):
        if idx.ndim != 2 or idx.shape[1] > self.cfg.block_size:
            raise ValueError("Input must be [batch, time] within configured context")
        t = idx.shape[1]
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(t, device=idx.device))
        if self.blocks is not None:
            for block in self.blocks:
                if self.cfg.backward_mode == "checkpointed" and self.training and torch.is_grad_enabled():
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
        else:
            x = self.stack(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        if targets.shape != idx.shape:
            raise ValueError("Targets must match input shape")
        loss_logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
        losses = F.cross_entropy(
            loss_logits.flatten(0, 1), targets.flatten(), ignore_index=-100,
            reduction="none" if return_token_losses else "sum",
        )
        return losses.view_as(targets) if return_token_losses else losses


def count_unique_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
