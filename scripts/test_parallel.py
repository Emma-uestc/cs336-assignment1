# test_parallel.py

from cs336_basics.train_bpe_mp import BPETrainer_MP

file_path = "tests/fixtures/tinystories_sample_5M.txt"
special_tokens = ["<|endoftext|>"]

trainer = BPETrainer_MP()

# 1. 跑单线程版
result_single = trainer._pretokenize_single(file_path, special_tokens)

# 2. 跑并行版（强制用 4 个进程）
result_parallel = trainer.pretokenize(file_path, special_tokens, num_workers=4)

# 3. 对比结果必须完全一致
assert result_single == result_parallel, "结果不一致！"
print(f"验证通过：共 {len(result_single)} 个不同词")
print(f"出现次数最多的前5个词：")
top5 = sorted(result_single.items(), key=lambda x: -x[1])[:5]
for word, count in top5:
    print(f"  {repr(word)}: {count}")