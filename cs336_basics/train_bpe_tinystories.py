import os
import sys
import psutil
from pathlib import Path
import json
import time
import cProfile
import argparse

from tests.common import gpt2_bytes_to_unicode


# Project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# try:
#     # from cs336_basics import train_bpe
#     from cs336_basics import train_bpe_optimizer
# except ImportError:
#     print("Error: Could not import train_bpe.Please run from project root or cs336_basics directory.")
#     sys.exit(1)

from cs336_basics import train_bpe_mp
from cs336_basics import train_bpe_mp_optimize
from cs336_basics import train_bpe_final


# get memory usage
def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# save vocab and merges to disk
from tests.common import gpt2_bytes_to_unicode

def save_vocab_and_merges(vocab, merges, output_dir='results'):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    byte_encoder = gpt2_bytes_to_unicode()  # {int: str}，each bytes map to printable string

    # Save vocab：each token's bytes covert to pritntable string 
    vocab_str = {}
    for idx, token_bytes in vocab.items():
        vocab_str[idx] = ''.join(byte_encoder[b] for b in token_bytes)

    with open(f'{output_dir}/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=2)

    # Save merges：two tokens are separated by space, and each byte is converted to printable string.
    with open(f'{output_dir}/merges.txt', 'w', encoding='utf-8') as f:
        for p1, p2 in merges:
            t1 = ''.join(byte_encoder[b] for b in p1)
            t2 = ''.join(byte_encoder[b] for b in p2)
            f.write(f'{t1} {t2}\n')


# main function
def run_training(input_path, vocab_size, special_tokens, output_dir='results'):
    """Run training"""
    # record the initial memory usage before training
    print(f'Initial Memory: {get_memory_usage_mb():.2f} MB')

    # Initialize the BPE Trainer instance
    # trainer = train_bpe.BPETrainer() # simple version
    # trainer = train_bpe_optimizer.BPETrainerOptimizer() # optimized version
    # trainer = train_bpe_mp.BPETrainer_MP()
    # trainer = train_bpe_mp_optimize.BPETrainer_MP_Optimizer()
    trainer = train_bpe_final.BPETrainer_Final()

    # start training and record the time and memory usage
    start_time = time.time()
    print(f'Starting training on {input_path}...')
    vocab,merges = trainer.train(input_path, vocab_size, special_tokens)
    end_time = time.time()
    duration = end_time - start_time
    peak_memory = get_memory_usage_mb()
    print('-' * 100)
    print('Training Complete.')
    print(f'Time Taken: {duration:.2f} seconds ({duration/60:.2f} minutes)')
    print(f'Final Memory: {peak_memory:.2f} MB')
    print('-' * 100)
    save_vocab_and_merges(vocab, merges, output_dir)

    # Output Statistics information
    print("\n=== Statistics (Problem b) ===")
    # 1. Longest token
    longest_token_bytes = max(vocab.values(), key=len)
    try:
        longest_token_str = longest_token_bytes.decode('utf-8')
    except:
        longest_token_str = str(longest_token_bytes)

    print(f"Longest Token: {longest_token_str!r}")
    print(f"Length in bytes: {len(longest_token_bytes)}")

    # 2. Most frequent token (approximate, based on merge priority if we tracked it, 
    # but here we can just say the last merged token was the most frequent *at that step*)
    # The assignment asks for "most frequent token in the dataset"? 
    # Usually BPE doesn't keep full frequency counts of final vocab unless we re-tokenize.
    # We will just print the last merge which represents the most frequent pair remaining.
    print(f"Total Merges: {len(merges)}")
    print('-' * 100)



def main():
    parser = argparse.ArgumentParser(description="Train BPE with profiling")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input dataset")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Target vocabulary size")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile")
    
    args = parser.parse_args()
    
    special_tokens = ["<|endoftext|>"]
    
    if not os.path.exists(args.input_path):
        print(f"Error: input file not found: {args.input_path}")
        sys.exit(1)

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        run_training(args.input_path, args.vocab_size, special_tokens, args.output_dir)
        profiler.disable()
        profiler.dump_stats("training_owt_final.prof")
    else:
        run_training(args.input_path, args.vocab_size, special_tokens, args.output_dir)

if __name__ == "__main__":
    main()
