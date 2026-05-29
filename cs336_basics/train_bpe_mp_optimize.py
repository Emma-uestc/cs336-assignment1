"""
BPE Trainer with Multiprocessing Optimization
OPtimize the communication between the main process and the worker processes 
using temporary files.
"""


from typing import List, Dict, Tuple
from collections import defaultdict
import os
import regex as re
import multiprocessing as mp

import tempfile
import pickle

N_BYTES = 256
"""
find_chunks_boundaries and _pretokenize_chunk must be as top level functions,
defined outside the class
because multiprocessing passes arguments to the child processes by pickling the functions
and class method cannot be pickled.
"""

pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# define the functionoutside the class, because multiprocessing needs to be able to pickle the funcion
def find_chunks_boundaries(file_path: str, num_chunks: int, special_token: str) -> List[int]:
    """
    Find the spliting boundaries of the file into num_chunks.
    split by front of special token, return num_chunks + 1 boundaries
    eg: num_chunks = 4, return [0, b1, b2, b3, file_size]
    Args:
        file_path: the path of the file to be pretokenized
        num_chunks: the number of chunks to split the file into
        special_token: the special token to split the file into chunks
    Returns:
        A list of byte offsets of the boundaries of the chunks
    """

    file_size = os.path.getsize(file_path)
    chunk_size = file_size // num_chunks
    boundaries = [0] # first boundary position is 0

    token_bytes = special_token.encode('utf-8')

    with open(file_path, 'rb') as f:
        for i in range(1, num_chunks):
            position = i * chunk_size
            f.seek(position)
            # start at position,read 4KB buffer
            buffer = f.read(4096)
            idx = buffer.find(token_bytes)
            if idx != -1:
                boundaries.append(position + idx)
            else:
                boundaries.append(position)
    boundaries.append(file_size)
    return boundaries

# Optimize: save child process result in temp file, main process read data from temp file
def _pretokenize_chunk(args: Tuple) -> str:
    """
    Process a chunk of text in a child process
    each child process handles its own chunk text segments [start, end]
    why use tuple?
    because pool.map can only pass one argument to the worker function
    """

    # get parameters from tuple
    file_path, start, end, special_tokens = args

    with open(file_path, 'r', encoding='utf-8') as f:
        f.seek(start)
        text = f.read(end - start)

    compiled_pattern = re.compile(pattern)
    word_counts = defaultdict(int)

    if special_tokens:
        escape_special_tokens = "|".join(re.escape(t) for t in special_tokens)
        compiled_special = re.compile(escape_special_tokens)
        chunks = compiled_special.split(text)
    else:
        chunks = [text]
    
    for chunk in chunks:
        for match in compiled_pattern.finditer(chunk):
            word_counts[match.group(0)] += 1

    # Optimize: add the 
    # add temporary file to save the word_counts
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    pickle.dump(dict(word_counts), tmp)
    tmp.close()
    return tmp.name # pass the file path to main process

   

class BPETrainer_MP_Optimizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []

    def train(self,
        input_path: str,
        vocab_size: int,
        special_tokens: List[str]
    ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        Args:
            input_path: str, the path of corpus file
            vocab_size: int, the size of the vocaulary
            special_tokens: List[str]
        Returns:
            vocabulary: Dict[int, bytes], the vocabulary of the BPE Trainer
        """
        # 1. Initialize the vocabulary
        self.vocab = {i: bytes([i]) for i in range(N_BYTES)}
        size = N_BYTES
        for token in special_tokens:
            self.vocab[size] = token.encode('utf-8')
            size += 1

        # 2. Pretokenize
        word_counts = self.pretokenize(input_path, special_tokens)

        # 3. Initialize the word encodings
        word_encodings = {word: list(word.encode('utf-8')) for word in word_counts}

        #***********************  Optimize 2 ***********************************************
        # create a index to query the words that include the pair
        # remian a map to record 
        pair_to_words = defaultdict(set)
        for word in word_encodings:
            encoding = word_encodings[word]
            for i in range(len(encoding) - 1):
                pair = (encoding[i], encoding[i+1])
                pair_to_words[pair].add(word)
        #*********************** Optimize 2 ************************************************

        # 4. BPE training loop
        num_merges = vocab_size - size
        pair_counts = defaultdict(int,self.count_pairs(word_counts, word_encodings))

        for merge_idx in range(num_merges):
            if not pair_counts:
                print('No more pairs to be merges, quit.')
                break
            # a. Find the pair with the highest frequency to be merged
            merge_pair = max(pair_counts,
            key=lambda x: (pair_counts[x], self.vocab[x[0]], self.vocab[x[1]])
            )

            token_id = size
            # Optimize 2: query the pair_to_words to get the affected words
            affected_words = pair_to_words.get(merge_pair, set()).copy()
            # update the pair_counts for the affected words
            for word in affected_words:
                old_encoding = word_encodings[word]
                new_encoding = self.merge_encoding(old_encoding, merge_pair, token_id)
                count = word_counts[word]

                # reduce from old_encoding and add to the new_encoding
                for i in range(len(old_encoding) - 1):
                    old_pair = (old_encoding[i], old_encoding[i+1])
                    pair_counts[old_pair] -= count
                    pair_to_words[old_pair].discard(word)

                for i in range(len(new_encoding) - 1):
                    new_pair = (new_encoding[i], new_encoding[i+1])
                    pair_counts[new_pair] += count
                    pair_to_words[new_pair].add(word)
                
                word_encodings[word] = new_encoding
            # clear the pairs whose count is no more than 0
            del_keys = [k for k, v in pair_counts.items() if v <= 0]
            for k in del_keys:
                del pair_counts[k]

            # c. update the merges and vocab
            self.merges.append((self.vocab[merge_pair[0]], self.vocab[merge_pair[1]]))
            self.vocab[token_id] = self.vocab[merge_pair[0]] + self.vocab[merge_pair[1]]
            size += 1

        return self.vocab, self.merges


    def merge_encoding(self,
        encoding: List[int],
        merge_pair: Tuple[int, int],
        token_id: int
    ) -> List[int]:

        """
        Args:
            encoding: List[int], the encoding of the word
            merge_pair: Tuple[int, int], the pair to be merged
            token_id: int, the new token id
        Returns:
            new_encoding: List[int], the new encding of the word
        """
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
            word_counts: Dict[str, int],the frequency of each word
            word_encodings: Dict[str, List[int]], the encoding of each word
        Returns:
            pair_counts: Dict[Tuple[int, int], int], the frequency of each pair of adjacent tokens
        """

        pair_counts = defaultdict(int)
        for word, count in word_counts.items():
            encoding = word_encodings[word]
            for i in range(len(encoding) - 1):
                pair = (encoding[i], encoding[i+1])
                pair_counts[pair] += count
        return dict(pair_counts)


    def pretokenize(
        self,
        input_path: str,
        special_tokens: List[str],
        num_workers: int=None
    ) -> Dict[str, int]:

        """
        Multiprocessing version using ProcessPoolExecutor.
        Each child process handles one segment,and returns the word counts for that segment.
        Pre-tokenization using parallel processing.
        num workers usually number of CPU cores 
        if num_workers is None, use max(1, os.cpu_count())

        Steps:
        1. Find chunk boundaries (split on <|endoftext|>)
        2. Read each chunk from file
        3. Submit chunks to ProcessPoolExecutor
        4. Merge results from all workers

        Optimize base BPETrainer_MP: pass tmp file path instead of big dict to main process
        """  
        if num_workers is None:
            num_workers = mp.cpu_count()

        # if file is too small, use multiprocessing will be slower than single process
        file_size = os.path.getsize(input_path)
        if file_size < 50 * 1024 * 1024: # 50MB
            return self._pretokenize_single(input_path, special_tokens)

        # 1. Find chunk boundaries
        split_token = special_tokens[0] if special_tokens else "\n"
        boundaries = find_chunks_boundaries(input_path, num_workers, split_token)

        # 2. Construct child process arguments
        args_list = [
            (input_path, boundaries[i], boundaries[i+1], special_tokens)
            for i in range(len(boundaries) - 1)
            if boundaries[i] < boundaries[i+1]
        ]

        # 3. Start child process
        with mp.Pool(processes=num_workers) as pool:
            tmp_paths = pool.map(_pretokenize_chunk, args_list)

        # 4. Merge results from all workers
        # Optimize: read tmp file from each child process and delete the tmp file after reading
        merged_counts = defaultdict(int)
        for tmp_path in tmp_paths:
            with open(tmp_path, 'rb') as f:
                partial_counts = pickle.load(f)
            for word, count in partial_counts.items():
                merged_counts[word] += count

            os.unlink(tmp_path) # delete the tmp file

        return dict(merged_counts)

    def _pretokenize_single(self,
        input_path: str,
        special_tokens: List[str]
    ) -> Dict[str, int]:
        """
        Args:
            input_path: str, the path of corpus file
            special_tokens: List[str], the special tokens list
        Returns:
            word_counts: Dict[str, int], the frequency of each word exclude special tokens
        """

        word_counts = defaultdict(int)

        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Optimization 1: compile the regex pattern and special tokensoutside the loop 
        # used in the loop to avoid re-compiling the regex pattern and special tokens each time
        compiled_pattern = re.compile(pattern)
        if special_tokens:
            escape_special_tokens = "|".join(re.escape(t) for t in special_tokens)
            compiled_special_tokens = re.compile(escape_special_tokens)
            chunks = compiled_special_tokens.split(text)
        else:
            chunks = [text]

        for chunk in chunks:
            for match in compiled_pattern.finditer(chunk):
                word_counts[match.group(0)] += 1

        return dict(word_counts)

        