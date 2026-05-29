
# corpus = "low low low low low\nlower lower\nwidest widest widest\nnewest newest newest newest newest newest\n<|endoftext|>"
corpus ="""
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
from collections import defaultdict

word_counts = defaultdict(int)
for m in re.finditer(PAT, corpus):
    word_counts[m.group(0)] += 1

word_encodings = {word: list(word.encode('utf-8')) for word in word_counts}
total_words = len(word_encodings)


affected = sum(1 for encoding in word_encodings.values()
    if any(encoding[i] == 115 and encoding[i+1] == 116 for i in range(len(encoding) - 1))
)
print(f'Total words: {total_words}')
print(f'Number of affected words after first merge (s,t): {affected}, {affected/total_words:.1%}')





