"""
Implement a naïve BPE trainer
"""

N_BYTES = 256
vocab_size = 1024

from ctypes import sizeof
import regex as re
from collections import defaultdict
from typing import Dict, List, Tuple
import os

class BPETrainer_Naive:
    def __init__(self):
        self.pattern =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab: Dict[int, bytes] = {}
        self.merges: List[Tuple[bytes, bytes]] = []

    def train(self,
        input_path: str,
        vocab_size: int,
        special_tokens: List[str]
        ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        Args:
            input_path: str, the path of the input file
            vocab_size: int, the size of the vocabulary
        Returns:
            vocabulary: Dict[int, bytes], the vocabulary of the BPE tokenizer
        """
        # TODO 1. Initialize the vocabulary
        # basic vocabulary [0,255]
        self.vocab = {i: bytes([i]) for i in range(N_BYTES)}
        # add special tokens to vocabulary
        size = N_BYTES
        for token in special_tokens:
            self.vocab[size] = token.encode('utf-8')
            size += 1
        # TODO 2. Pre-tokenize the text
        word_counts = self.pretokenize(input_path, special_tokens)
        # TODO 3. Initialize the word encodings
        word_encodings = {word: list(word.encode('utf-8')) for word in word_counts}
        # TODO 4. BPE training loop
        
        num_merges = vocab_size - size
        for merge_idx in range(num_merges):
            # a. find the max count pair to be merged
            pair_counts = self.count_pairs(word_counts, word_encodings)
            if not pair_counts:
                print('No more pairs to merge, quit')
                break
            merge_pair = max(
                pair_counts,
                key=lambda x: (pair_counts[x], self.vocab[x[0]], self.vocab[x[1]])
            )
            
            
            max_count = pair_counts[merge_pair]
            # b. merge and update the word encodings
            token_id = size
            for word in word_encodings:
                new_encoding = self.merge_encoding(word_encodings[word], merge_pair, token_id)
                word_encodings[word] = new_encoding
            # c. update the merges
            self.merges.append((self.vocab[merge_pair[0]], self.vocab[merge_pair[1]]))
            # d. update vocabulary
            self.vocab[token_id] = self.vocab[merge_pair[0]] + self.vocab[merge_pair[1]]
            size += 1

        return self.vocab, self.merges

    def merge_encoding(self,
        encoding: List[int],
        merge_pair: Tuple[int, int],
        token_id: int
        ) -> List[int]:

        new_encoding = []
        p0, p1 = merge_pair

        i = 0
        while i < len(encoding):
            if i < len(encoding) - 1 and encoding[i] == p0 and encoding[i+1] == p1:
                new_encoding.append(token_id)
                i += 2
            else:
                new_encoding.append(encoding[i])
                i += 1
        return new_encoding

    def count_pairs(self,
        word_counts: Dict[str, int],
        word_encodings: Dict[str, List[int]]
        ) -> Dict[Tuple[int, int], int]:
        """
        Args:
            word_counts: the frequency of each word got from pretokenize
            word_encodings: the initial encoding
        Returns:
            pair_counts: the bytes pair counts
        """
        pair_counts = defaultdict(int)

        for word, count in word_counts.items():
            encoding = word_encodings[word]
            for i in range(len(encoding) - 1):
                pair = (encoding[i], encoding[i+1])
                pair_counts[pair] += count
        return dict(pair_counts)


    def pretokenize(self, input_path: str, special_tokens: List[str] = ["<|endoftext|>"]) -> Dict[str, int]:
        """
        Args:
            input_path: str, the path of the input file
            special_token: List[str]
        Returns:
            word_counts: Dict[str, int], the frequency of each word, exclude special tokens
        """

        word_counts = defaultdict(int)

        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if special_tokens:
            escape_special_tokens = "|".join(re.escape(t) for t in special_tokens)
            chunks = re.split(escape_special_tokens, text)
        else:
            chunks = [text]
        
        for chunk in chunks:
            for match in re.finditer(self.pattern, chunk):
                word_counts[match.group(0)] += 1
        return dict(word_counts)


# if __name__ == "__main__":
    

#     input_path = 'tests/fixtures/tinystories_sample.txt'
#     special_tokens = ["<|endoftext|>"]

#     if not os.path.exists(input_path):
#         raise FileNotFoundError(f'Input file not found: {input_path}')
#     else:
#         # print(f'Testint pretokenize ....')
#         trainer = BPETrainer_Naive()

#         word_counts = trainer.pretokenize(input_path, special_tokens)
#         # print(f'\nFound {len(word_counts)} unique words.')

#         # sorted_words_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
#         # print(f'Top 10 most frequent words:')
#         # for word, count in sorted_words_counts[:10]:
#         #     print(f'{word}: {count}')
#         # print(f"Testing initialize word encodings ....")
#         word_encodings = {word: list(word.encode('utf-8')) for word in word_counts}
#         # print(f'Word encodings: {word_encodings}')

#         pair_counts = trainer.count_pairs(word_counts, word_encodings)
#         # print(f'Pair counts: {pair_counts}')

#         vocabulary, merges = trainer.train(input_path, vocab_size, special_tokens)

#         print(f'Vocabulary: {vocabulary}')
#         print(f'Vocabulary size: {len(vocabulary)}')
#         print('-'*200)
#         print(f'Merges: {merges}')
        
        


