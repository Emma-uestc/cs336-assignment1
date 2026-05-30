"""
CS336 Assignment 1 — Tokenizer Analysis (Section 2.7, problems a–d)

Run from the project root:
    python -m cs336_basics.run_analysis

Or with multiprocessing for the heavy encoding step (recommended for OWT):
    python -m cs336_basics.run_analysis --encode --workers 8
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

# Make sure `from cs336_basics...` works whether or not the file is
# inside a package (so `python cs336_basics/run_analysis.py` still works).
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cs336_basics.tokenizer_optimizer import Tokenizer

# ── paths ────────────────────────────────────────────────────────────────────
TS_VOCAB = "tinystories_tokenizer/vocab.json"
TS_MERGES = "tinystories_tokenizer/merges.txt"
OWT_VOCAB = "owt_tokenizer/vocab.json"
OWT_MERGES = "owt_tokenizer/merges.txt"

TS_TRAIN = "data/TinyStoriesV2-GPT4-train.txt"
TS_VALID = "data/TinyStoriesV2-GPT4-valid.txt"
OWT_TRAIN = "data/owt_train.txt"
OWT_VALID = "data/owt_valid.txt"  # optional

TS_OUT_TRAIN = "data/ts_train_tokens.npy"
TS_OUT_VALID = "data/ts_valid_tokens.npy"
OWT_OUT_TRAIN = "data/owt_train_tokens.npy"
OWT_OUT_VALID = "data/owt_valid_tokens.npy"

SPECIAL_TOKEN = "<|endoftext|>"
SPECIAL_BYTES = SPECIAL_TOKEN.encode("utf-8")
N_SAMPLES = 10
RANDOM_SEED = 42

# Stream read buffer used everywhere we walk a big corpus.
STREAM_BUF_BYTES = 4 * 1024 * 1024  # 4 MB


# ── streaming document iterator ──────────────────────────────────────────────


def iter_documents(path: str, separator: bytes = SPECIAL_BYTES) -> Iterable[str]:
    """
    Yield one document at a time from a large file split by `separator` bytes.

    Memory-bounded: only holds a 4 MB read buffer + current carry. Never loads
    the whole file. Safe on 10 GB OWT with 12 GB RAM.
    """
    with open(path, "rb") as f:
        carry = b""
        while True:
            buf = f.read(STREAM_BUF_BYTES)
            if not buf:
                break
            carry += buf
            # split keeps everything before each separator as a document,
            # the last piece is partial (no separator yet), carry it.
            parts = carry.split(separator)
            for p in parts[:-1]:
                doc = p.decode("utf-8", errors="replace").strip()
                if doc:
                    yield doc
            carry = parts[-1]
        # flush trailing partial doc (no trailing separator)
        if carry:
            doc = carry.decode("utf-8", errors="replace").strip()
            if doc:
                yield doc


def reservoir_sample(stream: Iterable[str], k: int, seed: int) -> list[str]:
    """
    Algorithm R: uniform-random sample of `k` items from an unknown-length stream.
    One pass, memory = O(k). Perfect for "pick 10 docs from 10 GB corpus".
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


# ── (a)(b) compression ratio ────────────────────────────────────────────────


def compute_ratio(tok: Tokenizer, docs: list[str]) -> tuple[float, int, int]:
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tok.encode(d)) for d in docs)
    return total_bytes / total_tokens, total_bytes, total_tokens


def task_ab(ts_tok: Tokenizer, owt_tok: Tokenizer) -> dict:
    print("\n" + "─" * 64)
    print("(a) Sampling 10 documents from each corpus (streaming + reservoir)")
    print("─" * 64)

    print("  sampling TinyStories train …", flush=True)
    ts_sample = reservoir_sample(iter_documents(TS_TRAIN), N_SAMPLES, RANDOM_SEED)
    print(f"    got {len(ts_sample)} docs")

    print("  sampling OpenWebText train …", flush=True)
    owt_sample = reservoir_sample(iter_documents(OWT_TRAIN), N_SAMPLES, RANDOM_SEED)
    print(f"    got {len(owt_sample)} docs")

    ts_ratio, ts_b, ts_t = compute_ratio(ts_tok, ts_sample)
    owt_ratio, owt_b, owt_t = compute_ratio(owt_tok, owt_sample)

    print("\n(a) Compression ratio on the matched tokenizer:")
    print(f"    TinyStories tok ⨯ TinyStories docs : {ts_b:>10,} B / {ts_t:>8,} tok"
          f" = {ts_ratio:6.3f} B/tok")
    print(f"    OpenWebText tok ⨯ OpenWebText docs : {owt_b:>10,} B / {owt_t:>8,} tok"
          f" = {owt_ratio:6.3f} B/tok")

    # ── (b) cross-tokenize ───────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("(b) Cross-tokenize: TinyStories tok on OpenWebText docs")
    print("─" * 64)

    cross_ratio, cross_b, cross_t = compute_ratio(ts_tok, owt_sample)
    rel = owt_ratio / cross_ratio  # >1 means OWT tok is more efficient
    print(f"    TinyStories tok ⨯ OpenWebText docs : {cross_b:>10,} B / {cross_t:>8,} tok"
          f" = {cross_ratio:6.3f} B/tok")
    print(f"    OpenWebText tok was {rel:.2f}× more efficient on OWT docs "
          f"({(rel - 1) * 100:.0f}% fewer tokens for the same text).")

    return {
        "ts_ratio": ts_ratio,
        "owt_ratio": owt_ratio,
        "cross_ratio": cross_ratio,
        "ts_sample": ts_sample,
        "owt_sample": owt_sample,
    }


# ── (c) throughput benchmark ────────────────────────────────────────────────


def task_c(owt_tok: Tokenizer, owt_sample: list[str]) -> None:
    print("\n" + "─" * 64)
    print("(c) Throughput benchmark (single-thread)")
    print("─" * 64)

    # Build a ~1 MB benchmark string by repeating the OWT sample.
    bench = "".join(owt_sample)
    while len(bench.encode("utf-8")) < 1_000_000:
        bench = bench * 2
    bench_bytes = len(bench.encode("utf-8"))
    print(f"  benchmark text: {bench_bytes/1e6:.2f} MB")

    # Warm-up once (regex compile etc.), then 3 timed runs, take median.
    _ = owt_tok.encode(bench[: min(len(bench), 50_000)])
    times = []
    for trial in range(3):
        t0 = time.perf_counter()
        _ = owt_tok.encode(bench)
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    throughput = bench_bytes / median  # bytes per second
    print(f"  median encode time : {median*1000:7.1f} ms")
    print(f"  throughput         : {throughput/1e6:7.3f} MB/s")

    pile_bytes = 825 * 1024**3  # 825 GiB
    pile_sec = pile_bytes / throughput
    print(f"  → 825 GB Pile would take ≈ {pile_sec/3600:.1f} hours "
          f"({pile_sec/86400:.2f} days) single-thread.")


# ── (d) encode full datasets to uint16 .npy ──────────────────────────────────


def explain_uint16(ts_tok: Tokenizer, owt_tok: Tokenizer) -> None:
    print("\n" + "─" * 64)
    print("(d) Why uint16?")
    print("─" * 64)
    ts_max = max(ts_tok.vocab.keys())
    owt_max = max(owt_tok.vocab.keys())
    print(f"    uint8  max = 255       → too small for any of our vocabs")
    print(f"    uint16 max = 65,535    → fits TS  max_id={ts_max:,} ✓"
          f" and OWT max_id={owt_max:,} ✓")
    print(f"    uint32 max = 4.3 × 10⁹ → wastes 2× space vs uint16")
    print("    → uint16 is the smallest type that holds every token id,")
    print("      which halves disk + RAM + IO vs int32. For OWT train")
    print("      (~2.5 B tokens) this saves ~5 GB.")


def _encode_one_doc(args):
    """Worker for multiprocessing pool. Returns (np.uint16 array,)."""
    vocab_path, merges_path, doc = args
    # Lazy-load tokenizer once per worker via a module-level global.
    global _WORKER_TOK
    if "_WORKER_TOK" not in globals() or _WORKER_TOK is None:
        _WORKER_TOK = Tokenizer.from_file(vocab_path, merges_path, [SPECIAL_TOKEN])
    ids = _WORKER_TOK.encode(doc)
    # Re-attach the document separator at the end so concatenation is faithful.
    ids.append(_WORKER_TOK.bytes_to_id[SPECIAL_BYTES])
    return np.asarray(ids, dtype=np.uint16)


def encode_file_streaming(
    vocab_path: str,
    merges_path: str,
    in_path: str,
    out_path: str,
    workers: int = 1,
) -> None:
    """
    Stream documents from in_path, encode them (optionally with mp.Pool),
    and append to out_path in chunks so we never hold all tokens in RAM.
    """
    import multiprocessing as mp

    print(f"\n  encoding {in_path}")
    print(f"           → {out_path}  (workers={workers})", flush=True)
    if not os.path.exists(in_path):
        print(f"    SKIP: input not found")
        return

    t0 = time.perf_counter()
    n_bytes = os.path.getsize(in_path)
    n_tokens = 0

    # We append uint16 arrays directly to a raw .bin first, then wrap as .npy.
    bin_path = out_path + ".tmp"
    with open(bin_path, "wb") as fout:

        def doc_args():
            for doc in iter_documents(in_path):
                yield (vocab_path, merges_path, doc)

        if workers <= 1:
            tok = Tokenizer.from_file(vocab_path, merges_path, [SPECIAL_TOKEN])
            for doc in iter_documents(in_path):
                ids = tok.encode(doc)
                ids.append(tok.bytes_to_id[SPECIAL_BYTES])
                arr = np.asarray(ids, dtype=np.uint16)
                arr.tofile(fout)
                n_tokens += len(arr)
        else:
            with mp.Pool(processes=workers) as pool:
                # imap_unordered keeps memory bounded; chunksize tuned for
                # ~1000 docs per dispatch to amortize IPC cost.
                for arr in pool.imap_unordered(
                    _encode_one_doc, doc_args(), chunksize=64
                ):
                    arr.tofile(fout)
                    n_tokens += len(arr)
                    if n_tokens % 5_000_000 < 100_000:
                        elapsed = time.perf_counter() - t0
                        rate = n_tokens / elapsed
                        print(f"    progress: {n_tokens/1e6:.1f}M tokens, "
                              f"{rate/1e6:.2f} M tok/s", flush=True)

    # Convert raw bin → real .npy (cheap, just adds a header).
    arr = np.fromfile(bin_path, dtype=np.uint16)
    np.save(out_path, arr)
    os.unlink(bin_path)

    elapsed = time.perf_counter() - t0
    print(f"    done: {n_tokens:,} tokens, {elapsed/60:.1f} min, "
          f"{n_bytes/elapsed/1e6:.2f} MB/s (input)")


# ── main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode", action="store_true",
                        help="also encode full train/valid datasets to .npy (task d)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2),
                        help="multiprocessing workers for full-dataset encoding")
    args = parser.parse_args()

    print("=" * 64)
    print("CS336 Tokenizer Analysis  —  Section 2.7")
    print("=" * 64)

    print("\nLoading tokenizers …")
    ts_tok = Tokenizer.from_file(TS_VOCAB, TS_MERGES, [SPECIAL_TOKEN])
    owt_tok = Tokenizer.from_file(OWT_VOCAB, OWT_MERGES, [SPECIAL_TOKEN])
    print(f"  TS  vocab size : {len(ts_tok.vocab):,}")
    print(f"  OWT vocab size : {len(owt_tok.vocab):,}")

    res = task_ab(ts_tok, owt_tok)
    task_c(owt_tok, res["owt_sample"])
    explain_uint16(ts_tok, owt_tok)

    if args.encode:
        print("\n" + "─" * 64)
        print("(d) Encoding full datasets to uint16 .npy "
              f"(workers={args.workers})")
        print("─" * 64)
        # smallest first so user gets feedback fast
        encode_file_streaming(TS_VOCAB, TS_MERGES, TS_VALID, TS_OUT_VALID,
                              workers=args.workers)
        if os.path.exists(OWT_VALID):
            encode_file_streaming(OWT_VOCAB, OWT_MERGES, OWT_VALID, OWT_OUT_VALID,
                                  workers=args.workers)
        encode_file_streaming(TS_VOCAB, TS_MERGES, TS_TRAIN, TS_OUT_TRAIN,
                              workers=args.workers)
        encode_file_streaming(OWT_VOCAB, OWT_MERGES, OWT_TRAIN, OWT_OUT_TRAIN,
                              workers=args.workers)
    else:
        print("\n  (skip full dataset encoding; pass --encode to run it)")

    print("\n" + "=" * 64)
    print("Done.")
    print("=" * 64)


if __name__ == "__main__":
    main()
