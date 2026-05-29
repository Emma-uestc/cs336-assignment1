# CS336 Assignment 1 — BPE Tokenizer & Transformer Foundations

This repository contains my implementation and optimization study for Stanford CS336 Assignment 1.

The project focuses on building a Byte Pair Encoding (BPE) tokenizer from scratch and exploring performance bottlenecks through profiling and iterative optimization.

## Features

### BPE Tokenizer

- Naive BPE implementation
- Incremental optimization of merge operations
- Pretokenization pipeline
- Multiprocessing-based pretokenization
- Streamed chunk reading for large corpora

### Performance Engineering

Profiling-driven optimization using Python profiling tools.

Key explorations include:

- bottleneck identification
- regex optimization
- chunked file reading
- multiprocessing acceleration
- memory-performance tradeoffs

### Dataset

Experiments were conducted on:

- OpenWebText (OWT)
- TinyStories

(Training data is excluded from this repository.)

## Notes & Analysis

Detailed implementation notes, profiling results, and optimization reasoning are available in:

```txt
notes/