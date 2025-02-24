#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from jaxtyping import Array, Float, Int
from model import tf


@dataclasses.dataclass(frozen=True)
class MetaModelConfig:
    chunk_len: int = 12
    stride_len: int = 12
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
    metamodel_config: MetaModelConfig,
    X: Int[Array, "T"],
    Y: Int[Array, "T"],
    inference: bool = False,
    inference_token: Int[Array, "1"] | None = None,
) -> Float[Array, "T k"]:
    params = {
        "WE": WE,
        "WQ": WQ,
        "WK": WK,
        "WV": WV,
        "WO": WO,
        "W1": W1,
        "W2": W2,
        "W3": W3,
    }
    opt_state, optimizer = create_opt_state(metamodel_config, params)

    assert metamodel_config.stride_len == metamodel_config.chunk_len, (
        "Non-chunk operation not implemented"
    )
    inputs, targets = (
        [
            X[i : i + metamodel_config.chunk_len]
            for i in range(0, len(X), metamodel_config.stride_len)
        ],
        [
            Y[i : i + metamodel_config.chunk_len]
            for i in range(0, len(X), metamodel_config.stride_len)
        ],
    )
    if len(inputs[-1]) < metamodel_config.chunk_len:
        inputs, inputs_left = jnp.array(inputs[:-1]), inputs[-1]
        targets, _ = jnp.array(targets[:-1]), targets[-1]
    else:
        inputs, inputs_left = jnp.array(inputs), jnp.array((), dtype=jnp.int32)
        targets, _ = jnp.array(targets), jnp.array((), dtype=jnp.int32)

    def process_chunk(optimizer, params__opt_state, batch):
        """Process one chunk"""
        params, opt_state = params__opt_state
        inputs, targets = batch
        parameterization = lambda p: partial(
            tf,
            d_attn,
            p["WE"],
            p["WQ"],
            p["WK"],
            p["WV"],
            p["WO"],
            p["W1"],
            p["W2"],
            p["W3"],
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

    logits = rearrange(logits, "c s k -> (c s) k")
    final_model = partial(
        tf,
        d_attn,
        new_params["WE"],
        new_params["WQ"],
        new_params["WK"],
        new_params["WV"],
        new_params["WO"],
        new_params["W1"],
        new_params["W2"],
        new_params["W3"],
    )
    if inference:
        leftover = jnp.concatenate((inputs_left, inference_token))
    else:
        leftover = inputs_left
    last = final_model(leftover)
    logits = jnp.concatenate((logits, last))

    return logits
