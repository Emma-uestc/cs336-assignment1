"""
BPE Trainer Final Version
Use max-heap to find the most frequent pair
in each iteration of BPE training instead of using max() function 
"""


from itertools import count
from typing import List, Dict, Tuple
from collections import defaultdict
import os
import regex as re
import multiprocessing as mp

import tempfile
import pickle
import heapq

N_BYTES = 256


pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# fix 3: use _RevBytes to reverse the bytes comparison so a min-heap behaves like a max-heap for the lexicographic tiebreaker required by BPE
class _RevBytes:
    """
    Wrapper that reverses bytes comparison so a min-heap behaves
    like a max-heap for the lexicographic tiebreaker required by BPE.

    BPE tiebreak rule (matches `max(..., key=(count, vocab[p0], vocab[p1]))`):
        among pairs sharing the highest count, pick the one with the
        lexicographically GREATEST (vocab[p0], vocab[p1]).
    `heapq` is a min-heap, so we wrap bytes such that "less than" means
    the underlying bytes are actually greater.
    """
    __slots__ = ("b",)

    def __init__(self, b: bytes):
        self.b = b

    def __lt__(self, other: "_RevBytes") -> bool:
        return self.b > other.b

    def __eq__(self, other) -> bool:
        return isinstance(other, _RevBytes) and self.b == other.b

    def __hash__(self) -> int:
        return hash(self.b)

"""
find_chunks_boundaries and _pretokenize_chunk must be as top level functions,
defined outside the class
because multiprocessing passes arguments to the child processes by pickling the functions
and class method cannot be pickled.
"""
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

# Worker buffer size for streaming reads. ~64 MB raw bytes → at worst ~400 MB
# Python str after PEP 393 widening and split duplication, well within budget.
_WORKER_BUFFER_BYTES = 64 * 1024 * 1024


def _pretokenize_chunk(args: Tuple) -> str:
    """
    Streaming pretokenize for one [start, end] byte range.

    KEY DESIGN: never load the whole [start, end] range into memory. Instead,
    repeatedly read a ~64 MB buffer, find the LAST special-token boundary in
    that buffer, process everything up to that boundary, then discard the
    processed text and continue from the boundary. This caps peak RSS per
    worker regardless of how big the assigned range is.

    Why split on special tokens: pre-tokenization regex must not cross a
    special token (e.g., '<|endoftext|>') because special tokens are atomic.
    Splitting on them is safe and guarantees no word straddles a buffer edge.
    """
    file_path, start, end, special_tokens = args

    compiled_pattern = re.compile(pattern)
    word_counts: Dict[str, int] = defaultdict(int)

    # Pre-compile a regex that finds any special token occurrence in raw bytes.
    # We work on bytes here to avoid decoding the whole buffer if the special
    # tokens are ASCII (which '<|endoftext|>' is). For non-ASCII special
    # tokens this still works because we use the same byte representation.
    if special_tokens:
        special_byte_alts = b"|".join(re.escape(t.encode("utf-8")) for t in special_tokens)
        compiled_special_bytes = re.compile(special_byte_alts)
    else:
        compiled_special_bytes = None

    def _flush_segment(segment_text: str) -> None:
        """Run word-level regex on one piece of text and accumulate counts."""
        for match in compiled_pattern.finditer(segment_text):
            word_counts[match.group(0)] += 1

    with open(file_path, "rb") as f:
        f.seek(start)
        remaining = end - start
        carry = b""  # bytes left over from previous buffer (after last special-token boundary)

        while remaining > 0:
            to_read = min(_WORKER_BUFFER_BYTES, remaining)
            buf = f.read(to_read)
            if not buf:
                break
            remaining -= len(buf)
            data = carry + buf

            if compiled_special_bytes is not None and remaining > 0:
                # find the LAST special-token match in `data`; everything up to
                # its end can be safely processed now, the tail becomes carry.
                last_match = None
                for m in compiled_special_bytes.finditer(data):
                    last_match = m
                if last_match is not None:
                    safe_end = last_match.end()
                    processable = data[:safe_end]
                    carry = data[safe_end:]
                else:
                    # no special token in this window — we cannot be sure no
                    # word straddles the buffer end. Carry the whole thing.
                    # In the pathological case where a single segment between
                    # specials is larger than the buffer, we still need to
                    # process; do a safe split on whitespace as a fallback.
                    if len(data) > 4 * _WORKER_BUFFER_BYTES:
                        # safety net: split at last whitespace to avoid OOM
                        cut = data.rfind(b"\n")
                        if cut == -1:
                            cut = data.rfind(b" ")
                        if cut == -1:
                            cut = len(data) // 2
                        processable, carry = data[:cut], data[cut:]
                    else:
                        carry = data
                        continue
            else:
                # no more bytes to read (final flush) or no special tokens at all
                processable = data
                carry = b""

            # Decode the processable bytes and run the per-segment pipeline.
            text = processable.decode("utf-8", errors="replace")
            if compiled_special_bytes is not None:
                # Split text on special tokens (these are NOT part of any word).
                # We use a Python-string regex here because we already decoded.
                escape_special_tokens = "|".join(re.escape(t) for t in special_tokens)
                segments = re.split(escape_special_tokens, text)
            else:
                segments = [text]

            for seg in segments:
                if seg:
                    _flush_segment(seg)

            # Free references before next iteration so memory peak stays low.
            del data, processable, text, segments

        # Final flush of any leftover bytes.
        if carry:
            text = carry.decode("utf-8", errors="replace")
            if compiled_special_bytes is not None:
                escape_special_tokens = "|".join(re.escape(t) for t in special_tokens)
                segments = re.split(escape_special_tokens, text)
            else:
                segments = [text]
            for seg in segments:
                if seg:
                    _flush_segment(seg)

    # Persist result via tmp file so the main process doesn't need to receive
    # a large dict through the pipe (same optimization as before).
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    pickle.dump(dict(word_counts), tmp)
    tmp.close()
    return tmp.name

   

class BPETrainer_Final:
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


        # ************************** Optimize 4 **********************************************
        """
        Use max-heap to find the most frequent pair in each iteration of BPE training
        """
        heap = []
        # Initialize the heap, push all pairs into the heap.
        # Use _RevBytes so the min-heap acts as a max-heap on the bytes tiebreaker
        # (BPE tiebreak: greatest (vocab[p0], vocab[p1]) wins).
        for pair, count in pair_counts.items():
            heapq.heappush(heap, (
                -count,
                _RevBytes(self.vocab[pair[0]]),
                _RevBytes(self.vocab[pair[1]]),
                pair,
            ))
        
        # Iterate until the heap is empty or we have performed num_merges merges
        for merge_idx in range(num_merges):
            if not pair_counts:
                print('No more pairs to be merged, quit.')
                break

            merge_pair = None
            while heap:
                neg_count, _, _, candidate_pair = heap[0]   
                current_count = pair_counts.get(candidate_pair, 0)

                if current_count <= 0:
                    heapq.heappop(heap)    # this pair is no longer in pair_counts, discard,continue to find next pair
                    continue
                if -neg_count != current_count:
                    heapq.heappop(heap)    # this outof date, discard, continue 
                    continue

                # find the max frequency pair, pop and save
                heapq.heappop(heap)
                merge_pair = candidate_pair
                break

            if merge_pair is None:
                break       
        # ************************** Optimize 4 **********************************************
            # merge as before version
            token_id = size
            # ↓ fix 1 update vocab first，then heappush can access it normally
            self.vocab[token_id] = self.vocab[merge_pair[0]] + self.vocab[merge_pair[1]]
            affected_words = pair_to_words.get(merge_pair, set()).copy()

            # Track EVERY pair whose count changed this iteration (both increased
            # and decreased). After pair_counts stabilizes we re-push them all
            # with their final count. This is essential: when an old_pair's count
            # is only decreased, its stale heap entry will fail the lazy-deletion
            # check and be discarded; without re-pushing, the pair would be lost
            # from the heap even though it is still a valid candidate.
            changed_pairs = set()
            for word in affected_words:
                old_encoding = word_encodings[word]
                new_encoding = self.merge_encoding(old_encoding, merge_pair, token_id)
                count = word_counts[word]

                # reduce from old_encoding and add to the new_encoding
                for i in range(len(old_encoding) - 1):
                    old_pair = (old_encoding[i], old_encoding[i+1])
                    pair_counts[old_pair] -= count
                    pair_to_words[old_pair].discard(word)
                    changed_pairs.add(old_pair)

                for i in range(len(new_encoding) - 1):
                    new_pair = (new_encoding[i], new_encoding[i+1])
                    pair_counts[new_pair] += count
                    pair_to_words[new_pair].add(word)
                    changed_pairs.add(new_pair)
                word_encodings[word] = new_encoding
            # clear the pairs whose count is no more than 0
            del_keys = [k for k, v in pair_counts.items() if v <= 0]
            for k in del_keys:
                del pair_counts[k]
            # All affected words processed, pair_counts is stable. Push the
            # current count for every changed pair that is still valid.
            # fix 2 push changed pairs to heap
            for changed_pair in changed_pairs:
                final_count = pair_counts.get(changed_pair, 0)
                if final_count > 0:
                    heapq.heappush(heap, (
                        -final_count,
                        _RevBytes(self.vocab[changed_pair[0]]),
                        _RevBytes(self.vocab[changed_pair[1]]),
                        changed_pair,
                    ))
            # c. update the merges and vocab
            self.merges.append((self.vocab[merge_pair[0]], self.vocab[merge_pair[1]]))
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
        file_size = os.path.getsize(input_path)

        # if file is too small, use multiprocessing will be slower than single process
        if file_size < 50 * 1024 * 1024: # 50MB
            return self._pretokenize_single(input_path, special_tokens)

        if num_workers is None:
            # Streaming worker keeps its peak RSS around _WORKER_BUFFER_BYTES * ~6
            # (~400 MB for a 64 MB buffer) regardless of the assigned range size,
            # so we can cap N by available RAM / per-worker peak directly.
            import psutil
            available_mb = psutil.virtual_memory().available / 1024 / 1024
            per_worker_mb_estimate = (_WORKER_BUFFER_BYTES / 1024 / 1024) * 6
            cpu_count = mp.cpu_count()
            mem_based_workers = max(1, int(available_mb * 0.8 / per_worker_mb_estimate))
            num_workers = min(cpu_count, mem_based_workers)
            print(
                f"[pretokenize] available={available_mb:.0f} MB, "
                f"per_worker_peak_est={per_worker_mb_estimate:.0f} MB "
                f"→ using {num_workers} workers"
            )

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
      

