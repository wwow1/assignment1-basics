from __future__ import annotations
import multiprocessing
import os
from queue import PriorityQueue
import regex as re
from dataclasses import dataclass, field
from .pretokenization_example import find_chunk_boundaries

class WordStrNode:
  byte_pair: bytes
  prev: WordStrNode | None = None
  next: WordStrNode | None = None
  dummy_head: bool = False

  def __init__(self, byte_pair: bytes, dummy_head: bool = False):
    self.byte_pair = byte_pair
    self.prev = None
    self.next = None
    self.dummy_head = dummy_head
  
  def _remove_myself(self):
    if self.prev is not None:
      self.prev.next = self.next
    if self.next is not None:
      self.next.prev = self.prev

  def _valid(self) -> bool:
    return self.prev is not None and self.dummy_head == False

  def _set_byte_pair(self, byte_pair: bytes):
    self.byte_pair = byte_pair

  def _link_next(self, next_node: WordStrNode):
    self.next = next_node
    next_node.prev = self

  def _link_prev(self, prev_node: WordStrNode):
    self.prev = prev_node
    prev_node.next = self

@dataclass(order=False)
class PrioritizedItem:
    freq: int
    pair: tuple[bytes, bytes]

    def __lt__(self, other):
        # 1. Frequency comparison (smaller value means higher priority in Min-Heap)
        # Since we store -freq, smaller -freq means larger freq.
        if self.freq != other.freq:
            return self.freq > other.freq
        
        # 2. Tie-breaking: Lexicographical order (Reverse)
        # We want LARGER dictionary order to be popped first.
        # In Min-Heap, "smaller" is popped first.
        # So we say self is "smaller" if self.pair is LARGER.
        return self.pair > other.pair

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pre_tokenize_chunk(input_path: str, start: int, end: int, special_tokens: list[str]) -> tuple[dict[str, int], dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], list[str]]]:
    """Pre-tokenize a string into frequency dictionaries.
    Returns:
        local_word_freqs: dict[str, int] - word frequencies
        local_token_freqs: dict[tuple[bytes, bytes], int] - byte pair frequencies
        local_byte_pair2words: dict[tuple[bytes, bytes], list[str]] - mapping from byte pair to words
    """
    with open(input_path, "rb") as f:
      f.seek(start)
      chunk = f.read(end - start).decode("utf-8", errors="ignore")

    special_tokens_set = set(special_tokens)
    local_word_freqs = {}
    local_token_freqs = {}
    local_byte_pair2words = {}

    def _process_tokens(tokens):
        for token in tokens:
            # 1. Word frequency
            local_word_freqs[token] = local_word_freqs.get(token, 0) + 1
            
            # 2. Token (byte pair) frequency and mapping
            word_bytes = token.encode("utf-8")
            if len(word_bytes) < 2:
                continue
                
            for i in range(len(word_bytes) - 1):
                byte_pair = (word_bytes[i:i+1], word_bytes[i+1:i+2])
                local_token_freqs[byte_pair] = local_token_freqs.get(byte_pair, 0) + 1
                
                if byte_pair not in local_byte_pair2words:
                    local_byte_pair2words[byte_pair] = []
                # According to user, duplication is fine/expected if multiple occurrences
                local_byte_pair2words[byte_pair].append(token)

    if not special_tokens:
        tokens = re.findall(PAT, chunk)
        _process_tokens(tokens)
    else:
        special_pattern = "(" + "|".join(f"(?: ?{re.escape(t)})" for t in special_tokens) + ")"
        parts = re.split(special_pattern, chunk)
        
        for part in parts:
            if not part:
                continue
            stripped_part = part.lstrip(" ")
            if stripped_part in special_tokens_set:
                continue
            else:
                tokens = re.findall(PAT, part)
                _process_tokens(tokens)
            
    return local_word_freqs, local_token_freqs, local_byte_pair2words

def build_byte_pair_node(word_bytes: bytes) -> WordStrNode:
  root = WordStrNode(b"", dummy_head=True)
  curr = root
  for i in range(len(word_bytes)):
    byte_pair = word_bytes[i:i+1]
    next_node = WordStrNode(byte_pair)
    curr._link_next(next_node)
    next_node._link_prev(curr)
    curr = next_node
  return root

def find_neighbor_byte(two_byte_pair: tuple[bytes, bytes], node: WordStrNode) -> tuple[WordStrNode, WordStrNode, WordStrNode, WordStrNode]:
  """
  Find the neighbor byte-pair of the given byte-pair in the linked list.
  """
  assert node.dummy_head == True, "Node must be dummy head"
  first_pair = two_byte_pair[0]
  second_pair = two_byte_pair[1]
  curr = node
  while curr.next is not None and curr.next.next is not None:
    if curr.next.byte_pair == first_pair and curr.next.next.byte_pair == second_pair:
      return curr, curr.next, curr.next.next, curr.next.next.next
    curr = curr.next
  return None, None, None, None

class BPE:
    def __init__(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        
        # State variables
        self.token_freqs = {}
        self.byte_pair2words = {}
        self.word2freqs = {}
        self.word2node = {}
        self.vocab = {}
        self.merges = []
        self.pq = PriorityQueue()
        self.vocab_next_token_id = 0

    def _build_vocab_and_frequencies(self):
        """
        Chunks the input file, parallelizes pre-tokenization, and then counts byte pair frequencies 
        and builds indices.
        """
        special_tokens_set = set(self.special_tokens)
        with open(self.input_path, "rb") as f:
            # parallelize this by sending each start/end pair to a set of processes
            boundaries = find_chunk_boundaries(f, 8, b"<|endoftext|>")

        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((self.input_path, start, end, self.special_tokens))
        
        with multiprocessing.Pool() as pool:
            results = pool.starmap(_pre_tokenize_chunk, tasks)
        
        # Merge results from all chunks
        for local_word_freqs, local_token_freqs, local_byte_pair2words in results:
            # 1. Merge word frequencies
            for word, freq in local_word_freqs.items():
                self.word2freqs[word] = self.word2freqs.get(word, 0) + freq
            
            # 2. Merge token frequencies
            for bp, freq in local_token_freqs.items():
                self.token_freqs[bp] = self.token_freqs.get(bp, 0) + freq
                
            # 3. Merge byte_pair2words
            for bp, words in local_byte_pair2words.items():
                if bp not in self.byte_pair2words:
                    self.byte_pair2words[bp] = []
                self.byte_pair2words[bp].extend(words)

        # 4. Build word nodes (still needed for linked list operations)
        # We only need to build nodes for UNIQUE words found
        for word in self.word2freqs:
             self.word2node[word] = build_byte_pair_node(word.encode("utf-8"))

    def _initialize_vocab(self):
        """
        Initialize the vocabulary with all single bytes and special tokens.
        """
        # init the vocabulary with all single bytes
        self.vocab = {i: i.to_bytes(1, "big") for i in range(256)}
        self.vocab_next_token_id = 256
        for token in self.special_tokens:
            self.vocab[self.vocab_next_token_id] = token.encode("utf-8")
            self.vocab_next_token_id += 1

    def _update_frequencies_after_merge(
        self,
        two_byte_pair: tuple[bytes, bytes],
        merged_pair: bytes,
    ) -> None:
        """
        Update frequencies of neighboring byte pairs after a merge operation.
        """
        first_pair = two_byte_pair[0]
        second_pair = two_byte_pair[1]
        
        # Track frequency changes to apply in batch
        # Use a dict to accumulate changes: pair -> delta
        freq_deltas = {}

        for word in self.byte_pair2words[two_byte_pair]:
            pre_node, first_node, second_node, next_node = find_neighbor_byte(two_byte_pair, self.word2node[word])
            # If the node has already been merged (e.g. in overlapping pairs like '000' -> '00', '0'),
            # the structure is changed. We skip if first_node is None.
            if first_node is None:
                continue
            
            # pre_node at least is dummy head
            assert pre_node is not None
            
            word_freq = self.word2freqs[word]
            
            # Decrement the frequency of the pair being merged
            # because this instance of the pair is now consumed
            freq_deltas[two_byte_pair] = freq_deltas.get(two_byte_pair, 0) - word_freq

            if pre_node is not None and pre_node._valid():
                pre_byte_pair = (pre_node.byte_pair, merged_pair)
                old_pre_pair = (pre_node.byte_pair, first_pair)
                
                # Update byte_pair2words for new pair
                if pre_byte_pair not in self.byte_pair2words:
                    self.byte_pair2words[pre_byte_pair] = []
                self.byte_pair2words[pre_byte_pair].append(word)
                
                # Accumulate freq changes
                freq_deltas[pre_byte_pair] = freq_deltas.get(pre_byte_pair, 0) + word_freq
                freq_deltas[old_pre_pair] = freq_deltas.get(old_pre_pair, 0) - word_freq

            if next_node is not None and next_node._valid():
                next_byte_pair = (merged_pair, next_node.byte_pair)
                old_next_pair = (second_pair, next_node.byte_pair)
                
                # Update byte_pair2words for new pair
                if next_byte_pair not in self.byte_pair2words:
                    self.byte_pair2words[next_byte_pair] = []
                self.byte_pair2words[next_byte_pair].append(word)

                # Accumulate freq changes
                freq_deltas[next_byte_pair] = freq_deltas.get(next_byte_pair, 0) + word_freq
                freq_deltas[old_next_pair] = freq_deltas.get(old_next_pair, 0) - word_freq
          
            first_node._set_byte_pair(merged_pair)
            second_node._remove_myself()

        # Apply frequency changes and update PQ
        for pair, delta in freq_deltas.items():
            self.token_freqs[pair] = self.token_freqs.get(pair, 0) + delta
            # Push the NEW frequency to PQ (Lazy Update)
            # We don't remove the old one; we check validity when popping from PQ.
            self.pq.put(PrioritizedItem(self.token_freqs[pair], pair))

    def _perform_merge_iteration(self) -> None:
        """
        Perform a single merge iteration: get best pair, merge it, update vocab and frequencies.
        """
        # Get the best valid pair from PQ
        while True:
            assert self.pq.empty() == False, "Priority queue should not be empty"
            item = self.pq.get()
            freq, byte_pair = item.freq, item.pair
            
            # Lazy deletion check:
            # If the popped frequency matches the current real frequency, it's valid.
            # Otherwise, it's a stale entry (because we updated frequency but didn't remove old entry), so we skip it.
            if freq == self.token_freqs.get(byte_pair, 0):
                break
                
        first_pair = byte_pair[0]
        second_pair = byte_pair[1]
        merged_pair = first_pair + second_pair
        
        # Log the merge with timestamp
        import datetime
        if self.vocab_next_token_id % 100 == 0:
            print(f"[{datetime.datetime.now()}]: {self.vocab_next_token_id} round, Merged: {first_pair} + {second_pair} -> {merged_pair}")

        # Update merge list and vocab
        self.merges.append(byte_pair)
        self.vocab[self.vocab_next_token_id] = merged_pair
        self.vocab_next_token_id += 1

        # Update frequencies
        self._update_frequencies_after_merge(
            byte_pair, merged_pair
        )

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # Count the frequency of each byte pair(2Byte)
        self._build_vocab_and_frequencies()
            
        # pq keep track of byte-pair frequency
        # the priority is the negative frequency of the byte-pair
        for byte_pair, freq in self.token_freqs.items():
            self.pq.put(PrioritizedItem(freq, byte_pair))
        
        self._initialize_vocab()

        # start to merge
        while self.vocab_next_token_id < self.vocab_size:
            assert self.pq.empty() == False, "Priority queue should not be empty"
            self._perform_merge_iteration()
            
        return self.vocab, self.merges

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    bpe = BPE(input_path, vocab_size, special_tokens)
    return bpe.train()
