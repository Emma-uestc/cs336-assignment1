"""
Implement a BPE tokenizer that encodes text into tokens and decodes tokens back into text.
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

        # add special tokens into the vocab if not in the vocab
        if self.special_tokens:
            for token in self.special_tokens:
                token_bytes = token.encode('utf-8')
                if token_bytes not in self.bytes_to_id:
                    new_id = max(self.vocab.keys()) + 1
                    self.vocab[new_id] = token_bytes
                    self.bytes_to_id[token_bytes] = new_id

        # # compile the pattern and special tokens
        # self.compiled_pattern = re.compile(pattern)
        # if self.special_tokens:
        #     escape_special_tokens = "|".join(re.escape(t) for t in self.special_tokens)
        #     self.compiled_special = re.compile(f"({escape_special_tokens})")
        # else:
        #     self.compiled_special = None
        

    @classmethod
    def from_file(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None
    ):
        # Contruct reflected map for unicode to bytes
        # gpt2_bytes_to_unicode() reutrn {int: str}
        # we need to reflect the map tp {str, int}
        unicode_to_bytes = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        # Load file from disk
        # read vocab.json
        # 文件格式: {"0": "Ā", "1": "ā", ..., "257": "Ġt", ...}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # Convert unicode to bytes
        vocab = {}
        for k, v in raw.items():
            # k is int with string type, v is printable string
            # convert printable string to byte and construct byte to bytes
            token_bytes = bytes([unicode_to_bytes[b] for b in v])
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
                # convert printable string to byte
                p0 = bytes([unicode_to_bytes[b] for b in p0_str])
                p1 = bytes([unicode_to_bytes[b] for b in p1_str])

                merges.append((p0, p1))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        """
        encode the text into token ids
        """
        # pretokenize
        ids = []
        compiled_pattern = re.compile(pattern)
        if self.special_tokens:
            escape_special_tokens = "|".join(re.escape(t) for t in self.special_tokens)
            compiled_special = re.compile(f"({escape_special_tokens})")
            chunks = compiled_special.split(text)

        else:
            chunks = [text]

        for chunk in chunks:
            # if chunk is a special token, query the byte_to_id map directly, do not merge
            if chunk in self.special_tokens:
                ids.append(self.bytes_to_id[chunk.encode('utf-8')])
                continue
            # if chunk is not a special token,merge bytes
            for match in compiled_pattern.finditer(chunk):
                word = match.group(0)
                tokens = [bytes([b]) for b in word.encode('utf-8')]

                for p0, p1 in self.merges:
                    new_tokens = []
                    i = 0
                    while i < len(tokens):
                        if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i+1] == p1:
                            new_tokens.append(p0 + p1)
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    tokens = new_tokens
            ids.extend([self.bytes_to_id[token] for token in tokens])
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





