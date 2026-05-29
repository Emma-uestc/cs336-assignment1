"""
Implement a tokenizer from BPE trainer
optimized version base on the tokenizer.py
"""

from email import message_from_bytes
from typing import Dict, List,Tuple,Iterable
from tests.common import gpt2_bytes_to_unicode
import json
import regex as re


pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        """
        Args:
            vocab: the vocabulary of the tokenizer
            merges: the merges rules of tokenizer
            special_tokens: special tokens provided by uer
        """
        self.vocab = vocab                    # dict[int, bytes]
        self.merges = merges                  # list[tuple[bytes, bytes]]
        self.special_tokens = special_tokens or []
        # construct the bytes to id map, by reflected vocab
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        # Newly added: create a rank query table
        self._merge_rank = {pair: i for i, pair in enumerate(self.merges)}

        # add special tokens into the vocab if not in the vocab
        if self.special_tokens:
            for token in self.special_tokens:
                token_bytes = token.encode('utf-8')
                if token_bytes not in self.bytes_to_id:
                    new_id = max(self.vocab.keys()) + 1
                    self.vocab[new_id] = token_bytes
                    self.bytes_to_id[token_bytes] = new_id

    @classmethod
    def from_file(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None
    ):

        # Contruct reflected map for unicode to bytes
        # gpt2_bytes_to_unicode() return {int: str}
        # we need to reflect the map tp {str, int}
        unicode_to_bytes = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        # Load file from disk
        # read vocab.json
        # 文件格式: {"0": "Ā", "1": "ā", ..., "257": "Ġt", ...}
        vocab = {}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # Convert unicode to bytes
        for k, v in raw.items():
            # k is int with string type, v is printable string
            # Convert printable string to byte and construct byte to bytes
            token_bytes = bytes(unicode_to_bytes[c] for c in v)
            vocab[int(k)] = token_bytes

        # read merges.txt
        # file format: each line: "Ġt he",two printable string separated by space
        # note: token itself may contain space(space is encoded as Ġ here),so split by the first space
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                # split by first space
                idx = line.index(' ')
                p0_str = line[:idx]
                p1_str = line[idx+1:]
                # Convert printable string to byte
                p0 = bytes(unicode_to_bytes[c] for c in p0_str)
                p1 = bytes(unicode_to_bytes[c] for c in p1_str)
                merges.append((p0, p1))
        return cls(vocab, merges, special_tokens)

    # Optimize: use the rank query table to speed up the merge
    def _bpe_encode_word(self, word_bytes: bytes) -> list[bytes]:
        encoding = [bytes([b]) for b in word_bytes]
        while len(encoding) >= 2:
            # Find the pair with the lowest rank to be merged
            pairs = [(encoding[i], encoding[i+1]) for i in range(len(encoding) - 1)]
            best = min(
                (p for p in pairs if p in self._merge_rank),
                key=lambda p: self._merge_rank[p],
                default=None,
            )
            if best is None:
                break
            p0, p1 = best
            new_encoding = []
            i = 0
            while i < len(encoding):
                if i < len(encoding) - 1 and encoding[i] == p0 and encoding[i+1] == p1:
                    new_encoding.append(p0 + p1)
                    i += 2
                else:
                    new_encoding.append(encoding[i])
                    i += 1
            encoding = new_encoding
        return encoding

    def encode(self, text: str) -> List[int]:
        """
        encode the text into token ids
        """
        ids = []
        compiled_pattern = re.compile(pattern)
        if self.special_tokens:
            # Sort by length descending so longer tokens are matched first
            # (regex alternation uses leftmost-alternative semantics, so without
            # this, "<|endoftext|><|endoftext|>" would be split into two
            # "<|endoftext|>" tokens when both are in special_tokens).
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escape_special_tokens = "|".join(re.escape(t) for t in sorted_specials)
            compiled_special = re.compile(f"({escape_special_tokens})")
            chunks = compiled_special.split(text)
        else:
            chunks = [text]
        for chunk in chunks:
            if chunk in self.special_tokens:
                ids.append(self.bytes_to_id[chunk.encode('utf-8')])
                continue
            for match in compiled_pattern.finditer(chunk):
                word = match.group(0)
                tokens = self._bpe_encode_word(word.encode('utf-8'))
                ids.extend(self.bytes_to_id[token] for token in tokens)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        encode the iterable into token ids
        generate token ids one by one
        do not store all token in memory
        """
        for text in iterable:
            ids = self.encode(text)
            yield from ids

    def decode(self, ids: List[int]) -> str:
        """
        decode token ids into text
        """
        token_bytes = b''.join(self.vocab[id] for id in ids)
        return token_bytes.decode('utf-8', errors='replace')