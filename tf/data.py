#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import jax
import jax.numpy as jnp


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


def generate_one_seq(word_len: int, seq_len: int, structure_coeff: float, key):
    key, subkey = jax.random.split(key)
    word = bit_array_to_bit_str(jax.random.randint(subkey, (word_len,), 0, 2))
    rate = 1 / (1 + structure_coeff * word_len)

    def generate_one_subseq(key):
        key, subkey = jax.random.split(key)
        next_word_loc = jax.random.geometric(key, rate).item() - 1
        noise = bit_array_to_bit_str(jax.random.randint(subkey, (next_word_loc,), 0, 2))
        return noise + word, next_word_loc

    def condition_fun(seq__key):
        seq, _, __ = seq__key
        return len(seq) < seq_len

    def body_fun(seq__locs__key):
        seq, locs, key = seq__locs__key
        key, subkey = jax.random.split(key)
        subseq, subloc = generate_one_subseq(subkey)
        seq += subseq
        locs += [subloc]
        return seq, locs, key

    # seq = jax.lax.while_loop(condition_fun, body_fun, (jnp.array([], dtype=jnp.int32), key))
    seq__locs__key = ("", [], key)
    while condition_fun(seq__locs__key):
        seq__locs__key = body_fun(seq__locs__key)

    seq, locs, key = seq__locs__key
    seq = seq[:seq_len]
    accumulated_locs = [locs[0]]
    for loc in locs[1:]:
        accumulated_locs += [accumulated_locs[-1] + word_len + loc]
    accumulated_locs = [loc for loc in accumulated_locs if loc < len(seq) - 1]

    return seq, word, accumulated_locs


def get_batch_of_seqs(
    key, word_len: int, seq_len: int, batch_size: int, structure_coeff: float = 1.0
):
    batch_word_locs = []
    batch_words = []
    batch_seqs = []
    for i in range(batch_size):
        # Geometric process to generate the sequence
        # NOTE: without doing some fairly serious magic, we can't pmap/etc this -- bc can't trace the random process
        # (need to do some concentration inequalities to guarantee reliability + maxsize stuff for tracing)
        key, subkey = jax.random.split(key)
        seq, word, word_locs = generate_one_seq(
            word_len, seq_len, structure_coeff, subkey
        )
        batch_seqs.append(seq)
        batch_words.append(word)
        batch_word_locs.append(word_locs)
    return batch_seqs, batch_words, batch_word_locs
