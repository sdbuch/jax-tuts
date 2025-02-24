#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports

import dataclasses
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int
from meta_model import MetaModelConfig, cross_entropy_loss, meta_tf
from model import ModelConfig, tf


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    seq_len: int = 48
    word_len: int = 6
    total_steps: int = 500
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


def bit_array_to_bit_str(bit_arr):
    return "".join(str(w.item()) for w in bit_arr)


def bit_str_to_bit_arr(bit_str):
    return jnp.array(list(map(int, list(bit_str))))


def demarcate_words(bit_str, word_locs, word_len):
    out = ""
    for i, char in enumerate(bit_str):
        if i in word_locs:
            out += f"|{char}"
        elif i in jax.tree.map(lambda x: x + word_len - 1, word_locs):
            out += f"{char}|"
        else:
            out += char
    return out


def get_batch_of_seqs(
    key, word_len: int, seq_len: int, batch_size: int, structure_coeff: float = 1.0
):
    batch_word_locs = []
    batch_words = []
    batch_seqs = []
    for i in range(batch_size):
        key, subkey = jax.random.split(key)
        word = bit_array_to_bit_str(jax.random.randint(subkey, (word_len,), 0, 2))
        # Poisson process to generate the sequence
        word_locs = []
        seq = ""
        while seq_len - len(seq) > 0:
            key, subkey = jax.random.split(key)
            next_word_loc = jax.random.poisson(
                subkey, structure_coeff * word_len
            ).item()
            key, subkey = jax.random.split(key)
            if not word_locs:
                if next_word_loc < seq_len - word_len + 1:
                    word_locs.append(next_word_loc)
                    noise = bit_array_to_bit_str(
                        jax.random.randint(subkey, (next_word_loc,), 0, 2)
                    )
                    seq += noise + word
                else:
                    noise = bit_array_to_bit_str(
                        jax.random.randint(subkey, (seq_len,), 0, 2)
                    )
                    seq += noise
            elif next_word_loc + word_locs[-1] + word_len < seq_len - word_len + 1:
                word_locs.append(word_locs[-1] + word_len + next_word_loc)
                noise = bit_array_to_bit_str(
                    jax.random.randint(subkey, (next_word_loc,), 0, 2)
                )
                seq += noise + word
            else:
                noise = bit_array_to_bit_str(
                    jax.random.randint(
                        subkey, (seq_len - word_locs[-1] - word_len,), 0, 2
                    )
                )
                seq += noise
        batch_seqs.append(seq)
        batch_words.append(word)
        batch_word_locs.append(word_locs)
    return batch_seqs, batch_words, batch_word_locs


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

    return {
        "WE": WE,
        "WQ": WQ,
        "WK": WK,
        "WV": WV,
        "WO": WO,
        "W1": W1,
        "W2": W2,
        "W3": W3,
    }


def create_train_state(model_config: ModelConfig, train_config: TrainConfig):
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
    key = jax.random.key(train_config.seed)
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
                params["WE"],
                params["WQ"],
                params["WK"],
                params["WV"],
                params["WO"],
                params["W1"],
                params["W2"],
                params["W3"],
                metamodel_config,
            )
            logits = jax.vmap(model)(inputs, targets)
        else:
            model = partial(
                tf,
                model_config.d_attn,
                params["WE"],
                params["WQ"],
                params["WK"],
                params["WV"],
                params["WO"],
                params["W1"],
                params["W2"],
                params["W3"],
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
    params, opt_state, optimizer = create_train_state(model_config, train_config)
    key = jax.random.key(train_config.seed)

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

        # Logging
        if step % train_config.log_rate == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    if train_config.overfit_batch:
        return params, (inputs, targets, seqs, words, word_locs)
    else:
        return params


@partial(jax.jit, static_argnames=["model_config", "metamodel_config"])
def compute_perplexity(
    params: dict,
    inputs: Int[Array, "B T"],
    targets: Int[Array, "B T"],
    model_config: ModelConfig,
    metamodel_config=None,
) -> Float[Array, ""]:
    """Compute perplexity on a batch of sequences."""
    if metamodel_config:
        model = partial(
            meta_tf,
            model_config.d_attn,
            params["WE"],
            params["WQ"],
            params["WK"],
            params["WV"],
            params["WO"],
            params["W1"],
            params["W2"],
            params["W3"],
            metamodel_config,
        )
        logits = jax.vmap(model)(inputs, targets)
    else:
        model = partial(
            tf,
            model_config.d_attn,
            params["WE"],
            params["WQ"],
            params["WK"],
            params["WV"],
            params["WO"],
            params["W1"],
            params["W2"],
            params["W3"],
        )
        logits = jax.vmap(model)(inputs)

    # Compute cross entropy loss
    loss = jax.vmap(lambda l, t: optax.softmax_cross_entropy_with_integer_labels(l, t))(
        logits, targets
    ).mean()

    # Convert to perplexity
    return jnp.exp(loss)


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
            params["WE"],
            params["WQ"],
            params["WK"],
            params["WV"],
            params["WO"],
            params["W1"],
            params["W2"],
            params["W3"],
            metamodel_config,
        )
        inputs, targets = sequence[:-1], sequence[1:]
        inference_token = sequence[-1][jnp.newaxis]
        logits = model(inputs, targets, inference=True, inference_token=inference_token)
    else:
        model = partial(
            tf,
            model_config.d_attn,
            params["WE"],
            params["WQ"],
            params["WK"],
            params["WV"],
            params["WO"],
            params["W1"],
            params["W2"],
            params["W3"],
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
) -> tuple[float, list[str]]:
    """Evaluate model on validation set and generate sample sequences."""
    # Generate validation set
    key = jax.random.key(val_config.seed + 1)  # Different seed from training
    seqs, words, word_locs = get_batch_of_seqs(
        key,
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
    perplexity = compute_perplexity(
        params, inputs, targets, model_config, metamodel_config=metamodel_config
    )

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

    return perplexity, samples


if __name__ == "__main__":
    # Example usage
    model_config = ModelConfig()
    train_config = TrainConfig(
        noise_coeff=0.0,
        batch_size=1,
        overfit_batch=True,
    )
    val_config = TrainConfig(
        batch_size=32, noise_coeff=train_config.noise_coeff
    )  # Use same config structure for validation
    metamodel_config = MetaModelConfig()
    # metamodel_config = None

    if train_config.overfit_batch:
        trained_params, memorized_stuff = train_model(
            model_config, train_config, metamodel_config=metamodel_config
        )
        perplexity, samples = evaluate_model(
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
        perplexity, samples = evaluate_model(
            trained_params, model_config, val_config, metamodel_config=metamodel_config
        )

    print(f"\nValidation Perplexity: {perplexity:.4f}\n")
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
