#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports

import dataclasses
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from baselines import oracle, reflector
from data import bit_str_to_bit_arr, demarcate_words, get_batch_of_seqs
from jaxtyping import Array, Float, Int
from meta_model import cross_entropy_loss, meta_tf
from model import ModelConfig, pack_params, tf, unpack_params


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    seq_len: int = 24
    word_len: int = 4
    total_steps: int = 2500
    warmup_steps: int = 250
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    log_rate: int = 100
    overfit_batch: bool = False
    noise_coeff: float = (
        1.0  # A larger value means generated sequences are more like noise
    )


def init_params(config: ModelConfig, key: Array):
    """Initialize transformer parameters with appropriate scaling."""
    keys = jax.random.split(key, 8)
    d = config.d_model

    # Xavier/Glorot initialization for embeddings
    scale_emb = jnp.sqrt(2.0 / (config.vocab_size + d))
    WE = jax.random.normal(keys[0], (config.vocab_size, d)) * scale_emb

    # Initialize attention matrices (scaled for QKV attention)
    scale_attn = 1.0  # We're using RMSnorm without Var, so we want to scale this up
    WQ = jax.random.normal(keys[1], (config.n_layers, d, d)) * scale_attn
    WK = jax.random.normal(keys[2], (config.n_layers, d, d)) * scale_attn
    WV = jax.random.normal(keys[3], (config.n_layers, d, d)) * scale_attn
    scale_attn = jnp.sqrt(2.0 / (2 * d))
    WO = jax.random.normal(keys[4], (config.n_layers, d, d)) * scale_attn

    # Initialize MLP matrices
    scale_mlp = jnp.sqrt(2.0 / (d + config.d_mlp))
    W1 = jax.random.normal(keys[5], (config.n_layers, config.d_mlp, d)) * scale_mlp
    W2 = jax.random.normal(keys[6], (config.n_layers, config.d_mlp, d)) * scale_mlp
    W3 = jax.random.normal(keys[7], (config.n_layers, d, config.d_mlp)) * scale_mlp

    return pack_params(WE, WQ, WK, WV, WO, W1, W2, W3)


def create_train_state(key, model_config: ModelConfig, train_config: TrainConfig):
    """Initialize training state with optimizer and parameters."""
    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=train_config.learning_rate,
        transition_steps=train_config.warmup_steps,
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=train_config.learning_rate,
        decay_steps=train_config.total_steps - train_config.warmup_steps,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[train_config.warmup_steps]
    )

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_config.grad_clip),
        optax.adamw(learning_rate=schedule_fn, weight_decay=train_config.weight_decay),
    )

    # Initialize parameters
    params = init_params(model_config, key)
    opt_state = optimizer.init(params)

    return params, opt_state, optimizer


@partial(jax.jit, static_argnames=["optimizer", "model_config", "metamodel_config"])
def train_step(
    params: dict,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: Tuple[Array, Array],
    model_config: ModelConfig,
    metamodel_config=None,
) -> Tuple[dict, optax.OptState, Float[Array, ""]]:
    """Single training step."""
    inputs, targets = batch

    def loss_fn(params):
        if metamodel_config:
            model = partial(
                meta_tf,
                model_config.d_attn,
                *unpack_params(params),
                metamodel_config,
            )
            logits = jax.vmap(model)(inputs, targets)
        else:
            model = partial(
                tf,
                model_config.d_attn,
                *unpack_params(params),
            )
            logits = jax.vmap(model)(inputs)
        return jax.vmap(cross_entropy_loss)(logits, targets).mean()

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


def prepare_batch(
    seqs: list[str], word_locs: list[list], word_len: int
) -> Tuple[Array, Array]:
    """Convert sequences to batched input and target tokens."""
    # Convert all sequences to arrays of integers
    batch_inputs = []
    batch_targets = []

    for seq in seqs:
        seq_array = jnp.array([int(x) for x in seq])
        batch_inputs.append(seq_array[:-1])
        batch_targets.append(seq_array[1:])

    # Stack into batched arrays
    inputs = jnp.stack(batch_inputs)
    targets = jnp.stack(batch_targets)
    return inputs, targets


def train_model(
    model_config: ModelConfig,
    train_config: TrainConfig,
    metamodel_config=None,
):
    """Main training loop."""
    # Initialize training state
    key = jax.random.key(train_config.seed)
    key, subkey = jax.random.split(key)
    params, opt_state, optimizer = create_train_state(
        subkey, model_config, train_config
    )

    if train_config.overfit_batch:
        key, subkey = jax.random.split(key)
        seqs, words, word_locs = get_batch_of_seqs(
            subkey,
            train_config.word_len,
            train_config.seq_len,
            train_config.batch_size,
            train_config.noise_coeff,
        )

        # Prepare batches
        inputs, targets = prepare_batch(seqs, word_locs, train_config.word_len)

    # Training loop
    losses = []
    for step in range(train_config.total_steps):
        if not train_config.overfit_batch:
            # Generate batch
            key, subkey = jax.random.split(key)
            seqs, words, word_locs = get_batch_of_seqs(
                subkey,
                train_config.word_len,
                train_config.seq_len,
                train_config.batch_size,
                train_config.noise_coeff,
            )
            # Prepare batches
            inputs, targets = prepare_batch(seqs, word_locs, train_config.word_len)

        # Training step
        if metamodel_config:
            params, opt_state, loss = train_step(
                params,
                opt_state,
                optimizer,
                (inputs, targets),
                model_config,
                metamodel_config,
            )
        else:
            params, opt_state, loss = train_step(
                params, opt_state, optimizer, (inputs, targets), model_config
            )
        losses.append(loss)

        # Logging
        if step % train_config.log_rate == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    # Write results
    fn = generate_filename(train_config, model_config, metamodel_config) + ".npy"
    jnp.save(fn, losses)

    if train_config.overfit_batch:
        return params, (inputs, targets, seqs, words, word_locs)
    else:
        return params


def generate_filename(train_config, model_config, metamodel_config=None):
    """
    Generate a detailed filename based on configuration parameters.

    Args:
        train_config: TrainConfig object
        model_config: ModelConfig object
        metamodel_config: MetaModelConfig object (optional)

    Returns:
        str: Formatted filename with parameter abbreviations
    """
    # Abbreviations for TrainConfig parameters
    tc_parts = [
        f"bs{train_config.batch_size}",
        f"sl{train_config.seq_len}",
        f"wl{train_config.word_len}",
        f"ts{train_config.total_steps}",
        f"ws{train_config.warmup_steps}",
        f"lr{train_config.learning_rate:.1e}",
        f"wd{train_config.weight_decay:.2f}",
        f"gc{train_config.grad_clip:.1f}",
        f"sd{train_config.seed}",
    ]

    # Add overfit_batch flag if True
    if train_config.overfit_batch:
        tc_parts.append("of")

    tc_parts.append(f"nc{train_config.noise_coeff:.1f}")

    # Abbreviations for ModelConfig parameters
    mc_parts = [
        f"dm{model_config.d_model}",
        f"da{model_config.d_attn}",
        f"dmlp{model_config.d_mlp}",
        f"nl{model_config.n_layers}",
    ]

    # Join all parts with underscores
    filename = f"TRAIN_{'_'.join(tc_parts)}_MODEL_{'_'.join(mc_parts)}"

    # Add MetaModelConfig parameters if provided
    if metamodel_config is not None:
        mmc_parts = [
            f"cl{metamodel_config.chunk_len}",
            f"stl{metamodel_config.stride_len}",
            f"ilr{metamodel_config.ilr:.1e}",
            f"gc{metamodel_config.grad_clip:.1f}",
        ]
        filename += f"_META_{'_'.join(mmc_parts)}"

    return filename


@partial(jax.jit, static_argnames=["model_config", "metamodel_config"])
def compute_perplexity(
    params: dict,
    inputs: Int[Array, "B T"],
    targets: Int[Array, "B T"],
    model_config: ModelConfig,
    metamodel_config=None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute perplexity on a batch of sequences."""
    if metamodel_config:
        model = partial(
            meta_tf,
            model_config.d_attn,
            *unpack_params(params),
            metamodel_config,
        )
        logits = jax.vmap(model)(inputs, targets)
    else:
        model = partial(
            tf,
            model_config.d_attn,
            *unpack_params(params),
        )
        logits = jax.vmap(model)(inputs)

    # Compute cross entropy loss
    loss = jax.vmap(lambda l, t: optax.softmax_cross_entropy_with_integer_labels(l, t))(
        logits, targets
    ).mean()
    preds = jnp.argmax(logits, axis=-1)
    acc = (preds == targets).mean()

    # Convert to perplexity
    return jnp.exp(loss), acc


@partial(jax.jit, static_argnames=["model_config", "temperature", "metamodel_config"])
def sample_next_token(
    key: jax.dtypes.prng_key,
    params: dict,
    sequence: Int[Array, "T"],
    model_config: ModelConfig,
    temperature: float = 1.0,
    metamodel_config=None,
) -> int:
    """Sample the next token given a sequence."""
    # Get logits from model
    if metamodel_config:
        model = partial(
            meta_tf,
            model_config.d_attn,
            *unpack_params(params),
            metamodel_config,
        )
        inputs, targets = sequence[:-1], sequence[1:]
        inference_token = sequence[-1][jnp.newaxis]
        logits = model(inputs, targets, inference=True, inference_token=inference_token)
    else:
        model = partial(
            tf,
            model_config.d_attn,
            *unpack_params(params),
        )
        logits = model(sequence)

    # Get final token logits and apply temperature
    final_logits = logits[-1] / temperature

    # Sample from the distribution
    return jax.random.categorical(key, final_logits)


def generate_sequence(
    key: jax.dtypes.prng_key,
    params: dict,
    model_config: ModelConfig,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    metamodel_config=None,
) -> str:
    """Generate a sequence given a prompt."""
    # Convert prompt to sequence of tokens
    sequence = jnp.array([int(x) for x in prompt])

    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        next_token = sample_next_token(
            key,
            params,
            sequence,
            model_config,
            temperature,
            metamodel_config=metamodel_config,
        )
        sequence = jnp.append(sequence, next_token)

    # Convert back to string
    return "".join(str(int(x)) for x in sequence)


def evaluate_model(
    params: dict,
    model_config: ModelConfig,
    val_config: TrainConfig,
    n_sequences: int = 100,
    override_batch=None,
    metamodel_config=None,
) -> tuple[float, list[str], dict]:
    """Evaluate model on validation set and generate sample sequences."""
    # Generate validation set
    key = jax.random.key(val_config.seed + 1)  # Different seed from training
    key, subkey = jax.random.split(key)
    seqs, words, word_locs = get_batch_of_seqs(
        subkey,
        val_config.word_len,
        val_config.seq_len,
        n_sequences,
        train_config.noise_coeff,
    )

    # Prepare validation data
    if override_batch:
        inputs, targets, seqs, words, word_locs = override_batch
    else:
        inputs, targets = prepare_batch(seqs, word_locs, val_config.word_len)

    # Compute perplexity
    perplexity, model_acc = compute_perplexity(
        params, inputs, targets, model_config, metamodel_config=metamodel_config
    )

    # Evaluate baselines
    oracle_fn = partial(oracle, val_config.seq_len)
    reflector_fn = partial(reflector, val_config.word_len)
    pad_max = max(map(len, word_locs))
    oracle_preds = jax.vmap(oracle_fn)(
        jnp.array(jax.tree.map(bit_str_to_bit_arr, words)),
        jnp.array(
            jax.tree.map(
                lambda x, pad_size=pad_max: jnp.pad(
                    x, pad_width=(0, pad_size - len(x)), mode="edge"
                ),
                word_locs,
            )
        ),
    )
    # This is very hard to make compatible with jax transforms
    reflector_preds = jnp.array([reflector_fn(input) for input in inputs])
    oracle_acc = (oracle_preds == targets).mean()
    reflector_acc = (reflector_preds == targets).mean()
    accs = {"model": model_acc, "oracle": oracle_acc, "algo baseline": reflector_acc}

    # Generate some sample sequences
    samples = []
    for i in range(min(5, n_sequences, len(seqs))):  # Generate 5 samples
        # Use first half of sequence as prompt
        prompt = seqs[i][: val_config.seq_len // 2]
        key, subkey = jax.random.split(key)
        generated = generate_sequence(
            subkey,
            params,
            model_config,
            prompt,
            val_config.seq_len // 2,  # Generate other half
            temperature=0.8,
            metamodel_config=metamodel_config,
        )
        samples.append(
            {
                "prompt": prompt,
                "generated": generated,
                "original": seqs[i],
                "word": words[i],
                "word_locations": word_locs[i],
            }
        )

    return perplexity, samples, accs


if __name__ == "__main__":
    # Example usage
    model_config = ModelConfig(
        n_layers=12,
        d_model=768,
        d_attn=48,
        d_mlp=2048,
    )

    train_config = TrainConfig(
        noise_coeff=0.5,
        seq_len=512,
        word_len=16,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-2,
        total_steps=5000,
        warmup_steps=500,
        # batch_size=1,
        # overfit_batch=True,
        # total_steps=500,
    )
    val_config = train_config
    # metamodel_config = MetaModelConfig(
    #     ilr=1e-3,
    # )
    metamodel_config = None

    if train_config.overfit_batch:
        trained_params, memorized_stuff = train_model(
            model_config, train_config, metamodel_config=metamodel_config
        )
        perplexity, samples, accs = evaluate_model(
            trained_params,
            model_config,
            val_config,
            override_batch=memorized_stuff,
            metamodel_config=metamodel_config,
        )
    else:
        trained_params = train_model(
            model_config, train_config, metamodel_config=metamodel_config
        )
        perplexity, samples, accs = evaluate_model(
            trained_params, model_config, val_config, metamodel_config=metamodel_config
        )

    print(f"\nValidation Perplexity: {perplexity:.4f}\n")
    print("\nValidation Accuracies")
    for k, v in accs.items():
        print(f"{k}: {v}")
    print("Sample Generations:")
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"Word: {sample['word']}")
        print(
            f"Prompt:     {demarcate_words(sample['prompt'], sample['word_locations'], val_config.word_len)}"
        )
        print(f"Generated:  {sample['generated'][len(sample['prompt']) :]}")
        print(f"Original:   {sample['original'][len(sample['prompt']) :]}")
#     print(demarcate_words(seqs[0], word_locs[0], 4))
