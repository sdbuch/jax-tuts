#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from jaxtyping import Array, Float, Int
from model import pack_params, tf, unpack_params


@dataclasses.dataclass(frozen=True)
class MetaModelConfig:
    chunk_len: int = 8
    stride_len: int = 1
    ilr: float = 1e-1
    grad_clip: float = 1.0


def cross_entropy_loss(
    logits: Float[Array, "T V"],
    targets: Int[Array, "T"],
) -> Float[Array, ""]:
    """Compute cross entropy loss for next-token prediction."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


def create_opt_state(metamodel_config: MetaModelConfig, params: dict):
    """Initialize training state with SGD optimizer and parameters."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(metamodel_config.grad_clip),
        optax.sgd(learning_rate=metamodel_config.ilr),
    )

    opt_state = optimizer.init(params)
    return opt_state, optimizer


def meta_tf(
    d_attn: int,
    WE: Float[Array, "k d"],
    WQ: Float[Array, "L d d"],
    WK: Float[Array, "L d d"],
    WV: Float[Array, "L d d"],
    WO: Float[Array, "L d d"],
    W1: Float[Array, "L d_mlp d"],
    W2: Float[Array, "L d_mlp d"],
    W3: Float[Array, "L d d_mlp"],
    mc: MetaModelConfig,
    X: Int[Array, "T"],
    Y: Int[Array, "T"],
    inference: bool = False,
    inference_token: Int[Array, "1"] | None = None,
) -> Float[Array, "T k"]:
    params = pack_params(WE, WQ, WK, WV, WO, W1, W2, W3)
    opt_state, optimizer = create_opt_state(mc, params)

    assert mc.stride_len == mc.chunk_len or mc.stride_len == 1, (
        "Non-chunk/non-sliding operation not permitted"
    )
    inputs, targets = (
        [
            X[i : i + mc.chunk_len]
            for i in range(0, len(X) - mc.chunk_len + 1, mc.stride_len)
        ],
        [
            Y[i : i + mc.chunk_len]
            for i in range(0, len(X) - mc.chunk_len + 1, mc.stride_len)
        ],
    )
    if len(X) < mc.chunk_len:
        # Sliding window operation, and the input is smaller than the chunk length
        inputs, targets = (
            jnp.array((inputs,), dtype=jnp.int32),
            jnp.array((targets,), dtype=jnp.int32),
        )
        inputs_left = X
    elif len(inputs[-1]) < mc.chunk_len:
        # Chunk operation, and we have some leftover
        inputs, inputs_left = jnp.array(inputs[:-1]), inputs[-1]
        targets, _ = jnp.array(targets[:-1]), targets[-1]
    else:
        # Either operation, no leftover
        inputs, inputs_left = jnp.array(inputs), jnp.array((), dtype=jnp.int32)
        targets, _ = jnp.array(targets), jnp.array((), dtype=jnp.int32)

    def process_chunk(optimizer, params__opt_state, batch):
        """Process one chunk"""
        params, opt_state = params__opt_state
        inputs, targets = batch
        parameterization = lambda p: partial(
            tf,
            d_attn,
            *unpack_params(p),
        )
        loss_fn = lambda p: cross_entropy_loss(
            parameterization(p)(inputs), targets
        ).mean()
        loss, grads = jax.value_and_grad(loss_fn)(params)
        logits = parameterization(params)(inputs)  # TODO: Wasting
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), logits

    chunk_fn = partial(process_chunk, optimizer)
    (new_params, new_opt_state), logits = jax.lax.scan(
        chunk_fn, (params, opt_state), (inputs, targets)
    )

    if mc.stride_len == 1:
        # Sliding mode: keep first, the rest predict 1 token
        if logits.shape[0] > 1:
            first, rest = logits[0], logits[1:]
            logits = jnp.concatenate((first, rest[:, -1, :]))
        else:
            logits = logits[0]
    else:
        # Chunk mode: un-chunk
        logits = rearrange(logits, "c s k -> (c s) k")
    final_model = partial(
        tf,
        d_attn,
        *unpack_params(new_params),
    )
    if inference:
        if mc.stride_len == 1:
            # SW mode: Put the inference tok on its SW
            if len(inputs_left) > 0:
                leftover = jnp.concatenate((inputs_left, inference_token))
            else:
                leftover = jnp.concatenate((inputs[-1, 1:], inference_token))
        else:
            # Chunk mode: make a new chunk out of what's left + the inference tok
            leftover = jnp.concatenate((inputs_left, inference_token))
    else:
        leftover = inputs_left
    last = final_model(leftover)
    logits = jnp.concatenate((logits, last))

    return logits
