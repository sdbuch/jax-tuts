#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, Int


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 2  # Binary sequences
    d_model: int = 96
    d_attn: int = 12
    d_mlp: int = 256
    n_layers: int = 4


def gated_mlp(
    W1: Float[Array, "d_mlp d"],
    W2: Float[Array, "d_mlp d"],
    W3: Float[Array, "d d_mlp"],
    x: Float[Array, "d"],
) -> Float[Array, "d"]:
    f = jax.nn.silu(W1 @ x)
    g = W2 @ x
    return W3 @ (f * g)


def attn(
    Q: Float[Array, "T d_attn"],
    K: Float[Array, "T d_attn"],
    V: Float[Array, "T d_attn"],
) -> Float[Array, "T d_attn"]:
    d_attn = Q.shape[1]
    A = Q @ K.T / jnp.sqrt(d_attn)
    causal_mask = jnp.tril(jnp.ones_like(A))
    A += jnp.where(causal_mask, 0, -jnp.inf)
    A = jax.nn.softmax(A)
    return A @ V


def multi_head_attn(
    d_attn: int,
    Q: Float[Array, "T d"],
    K: Float[Array, "T d"],
    V: Float[Array, "T d"],
    WO: Float[Array, "d d"],
) -> Float[Array, "T d"]:
    d = Q.shape[1]
    assert d % d_attn == 0, (
        f"Number of attention heads needs to divide model dimension ({d}, {d_attn})"
    )
    n_heads = d // d_attn
    Q, K, V = jax.tree.map(
        lambda x: rearrange(x, "T (h d_attn) -> h T d_attn", h=n_heads), (Q, K, V)
    )
    heads = jax.vmap(attn, in_axes=(0, 0, 0), out_axes=0)(
        Q, K, V
    )  # n_heads x T x d_attn
    attn_out = rearrange(heads, "h T d_attn -> T (h d_attn)")
    return attn_out @ WO.T


def multi_head_self_attn(
    d_attn: int,
    WQ: Float[Array, "d d"],
    WK: Float[Array, "d d"],
    WV: Float[Array, "d d"],
    WO: Float[Array, "d d"],
    X: Float[Array, "T d"],
) -> Float[Array, "T d"]:
    Q, K, V = X @ WQ.T, X @ WK.T, X @ WV.T
    return multi_head_attn(d_attn, Q, K, V, WO)


def rms_norm(X: jnp.ndarray, eps=1e-6):
    N = jnp.linalg.norm(X, axis=-1, keepdims=True) ** 2
    return X / jnp.sqrt(N + eps)


def block_MHSA(
    d_attn: int,
    WQ: Float[Array, "d d"],
    WK: Float[Array, "d d"],
    WV: Float[Array, "d d"],
    WO: Float[Array, "d d"],
    X: Float[Array, "T d"],
) -> Float[Array, "T d"]:
    mhsa = partial(multi_head_self_attn, d_attn, WQ, WK, WV, WO)
    return X + mhsa(rms_norm(X))


def block_MLP(
    W1: Float[Array, "d_mlp d"],
    W2: Float[Array, "d_mlp d"],
    W3: Float[Array, "d d_mlp"],
    X: Float[Array, "T d"],
) -> Float[Array, "T d"]:
    mlp = partial(gated_mlp, W1, W2, W3)
    return X + jax.vmap(mlp)(rms_norm(X))


def block_tf(
    d_attn: int,
    WQ: Float[Array, "d d"],
    WK: Float[Array, "d d"],
    WV: Float[Array, "d d"],
    WO: Float[Array, "d d"],
    W1: Float[Array, "d_mlp d"],
    W2: Float[Array, "d_mlp d"],
    W3: Float[Array, "d d_mlp"],
    X: Float[Array, "T d"],
) -> Float[Array, "T d"]:
    block_mhsa = partial(block_MHSA, d_attn, WQ, WK, WV, WO)
    block_mlp = partial(block_MLP, W1, W2, W3)
    return block_mlp(block_mhsa(X))


def tf(
    d_attn: int,
    WE: Float[Array, "k d"],
    WQ: Float[Array, "L d d"],
    WK: Float[Array, "L d d"],
    WV: Float[Array, "L d d"],
    WO: Float[Array, "L d d"],
    W1: Float[Array, "L d_mlp d"],
    W2: Float[Array, "L d_mlp d"],
    W3: Float[Array, "L d d_mlp"],
    X: Int[Array, "T"],
) -> Float[Array, "T k"]:
    embeddings = WE[X]  # T x d

    def block_fn(x, p):
        return block_tf(d_attn, *p, x), None

    out, _ = jax.lax.scan(block_fn, embeddings, (WQ, WK, WV, WO, W1, W2, W3))
    logits = out @ WE.T
    return logits


if __name__ == "__main__":
    L = 2
    d_model = 4
    d_attn = 2
    k = 2
    T = 10
    d_mlp = d_model
    WQ = []
    for i in range(L):
        WQ.append(jnp.eye(d_model))
    WQ = jnp.array(WQ)
    WK = WQ.copy()
    WV = WQ.copy()
    WO = WQ.copy()
    W1 = WQ.copy()
    W2 = WQ.copy()
    W3 = WQ.copy()
    WE = jnp.eye(d_model)[:k]
    x = "0011101010"
    X = jnp.array(list(map(int, list(x))))
    assert len(X) == T
    transformer = partial(tf, d_attn, WE, WQ, WK, WV, WO, W1, W2, W3)
    transformer(X)
