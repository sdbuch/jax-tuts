#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import jax
import jax.numpy as jnp
from data import bit_array_to_bit_str, bit_str_to_bit_arr


def oracle(seq_len, word, locs):
    """
    This baseline knows exactly where the secret words are, and predicts accordingly
    Supervision-only baseline (no inference!)

    We output the next-token predictions for the sequence
    """
    output = jnp.zeros((seq_len,), dtype=jnp.int32)
    for loc in locs:
        output = jax.lax.dynamic_update_slice(output, word, (loc,))
    return output[1:]


def reflector(word_len, seq):
    """
    This baseline estimates the parameters of the data, then generates accordingly
    Assumes we know the word length
    Supervision-only baseline (no inference!)

    seq: should be a token sequence
         we output the next token predictions for this sequence
    """

    # Estimate the secret word, using sqrt(len(seq)) tokens
    est_len = int(jnp.floor(len(seq) ** 0.5))
    assert est_len >= word_len, (
        "Need sqrt(seq_len) to be at least as large as the word_len"
    )

    seq_est, seq_pred = seq[:est_len], seq[est_len:]
    wgrams = get_ngrams(bit_array_to_bit_str(seq_est), word_len)
    word_est, word_est_occ = max(wgrams.items(), key=lambda x: x[1])
    word_est = bit_str_to_bit_arr(word_est)

    # # For the rest of the tokens, run a FSM
    # first_bit = jnp.array(word_est[0].item(), dtype=jnp.int32)
    # def scan_fn(pos, tok):
    #     """FSM implementation. But lacks a table (KMP)"""
    #     condition = tok != word_est[pos]
    #     pos_branch = lambda pos: (first_bit, first_bit)
    #     neg_branch = lambda pos: (
    #         (pos + 1) % len(word_est),
    #         word_est[(pos + 1) % len(word_est)],
    #     )
    #     return jax.lax.cond(condition, pos_branch, neg_branch, pos)
    # _, next_tok_pred = jax.lax.scan(scan_fn, first_bit, seq_pred)

    def scan_fn(window, tok):
        """
        Naive window-based matcher. Window must be len(word_est)-1 size.
        Issue: the matcher is perfect at lambda=0! But it's not noise robust.
        Noise before a word can lead to messed up outputs during the word.
        """
        ctx = jnp.concatenate((window, tok[jnp.newaxis]))
        matched_subctx = jnp.array(-1, dtype=jnp.int32)
        for offset in range(len(ctx)):
            predicate = jnp.all(ctx[-(1 + offset) :] == word_est[: (1 + offset)])
            matched_subctx = jax.lax.select(
                predicate, jnp.array(offset), matched_subctx
            )
        out = word_est[(matched_subctx + 1) % len(word_est)]
        return ctx[1:], out

    first_window = seq_est[-(len(word_est) - 1) :]
    _, next_tok_pred = jax.lax.scan(scan_fn, first_window, seq_pred)

    return jnp.concatenate((jnp.zeros_like(seq_est, dtype=jnp.int32), next_tok_pred))


def get_ngrams(seq, n):
    ngrams = {}
    for i in range(len(seq) - n):
        ngram = seq[i : i + n]
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    ngrams = dict(sorted(ngrams.items(), key=lambda item: item[1]))
    return ngrams
