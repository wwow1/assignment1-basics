from __future__ import annotations
import multiprocessing
from functools import partial
import regex as re
import heapq
from typing import Iterable, Iterator
from .pre_tokenization import pre_tokenize_string
from .word_node import WordStrNode
import multiprocessing
from functools import partial

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] = None
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        # Build inverse vocab for fast bytes -> id lookup
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        self.merges_ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r") as f:
            vocab = {int(line.split()[0]): line.split()[1].encode("utf-8") for line in f}
        
        with open(merges_filepath, "r") as f:
            merges = [line.split() for line in f]

        return cls(vocab, merges, special_tokens)

    def _init_linked_list_and_queue(self, word_bytes: bytes) -> tuple[WordStrNode, list]:
        """
        Builds the linked list of WordStrNodes from bytes and initializes the priority queue
        with mergeable pairs found during construction.
        """
        dummy = WordStrNode(b"", dummy_head=True)
        prev_node = dummy
        pq = []
        
        for i in range(len(word_bytes)):
            node = WordStrNode(word_bytes[i:i+1])
            prev_node._link_next(node)
            
            # Check merge with previous node (if previous is not dummy head)
            if not prev_node.dummy_head:
                pair = (prev_node.byte_pair, node.byte_pair)
                if pair in self.merges_ranks:
                    priority = self.merges_ranks[pair]
                    heapq.heappush(pq, (priority, id(prev_node), prev_node))
            
            prev_node = node
            
        return dummy, pq

    def _merge_byte_pairs(self, pq: list, dummy: WordStrNode):
        """
        Processes the priority queue to merge byte pairs in the linked list.
        """
        while pq:
            priority, _, node = heapq.heappop(pq)
            
            # Check validity: node must be valid and have a next node
            if not node._valid() or node.next is None:
                continue
            
            # Check if the pair is still the same (priority check)
            pair = (node.byte_pair, node.next.byte_pair)
            if self.merges_ranks.get(pair) != priority:
                continue
            
            # Merge
            next_node = node.next
            new_bytes = node.byte_pair + next_node.byte_pair
            node._set_byte_pair(new_bytes)
            next_node._remove_myself()
            
            # Check new neighbors
            # (node.prev, node)
            if node.prev and not node.prev.dummy_head:
                pair = (node.prev.byte_pair, node.byte_pair)
                if pair in self.merges_ranks:
                    priority = self.merges_ranks[pair]
                    heapq.heappush(pq, (priority, id(node.prev), node.prev))
            
            # (node, node.next)
            if node.next:
                pair = (node.byte_pair, node.next.byte_pair)
                if pair in self.merges_ranks:
                    priority = self.merges_ranks[pair]
                    heapq.heappush(pq, (priority, id(node), node))

    def _get_tokens_from_list(self, dummy: WordStrNode) -> list[int]:
        """
        Traverses the linked list to collect token IDs.
        """
        ids = []
        curr = dummy.next
        while curr:
            ids.append(self.inverse_vocab[curr.byte_pair])
            curr = curr.next
        return ids

    def _encode_chunk(self, word_bytes: bytes) -> list[int]:
        """
        Encodes a single chunk of bytes (a word) using BPE.
        """
        if not word_bytes:
            return []
            
        dummy, pq = self._init_linked_list_and_queue(word_bytes)
        self._merge_byte_pairs(pq, dummy)
        return self._get_tokens_from_list(dummy)

    def _encode_tokens(self, tokens: list[str]) -> list[int]:
        """
        Encodes a list of tokens into IDs.
        """
        encoded_ids = []
        for token in tokens:
            if token in self.special_tokens:
                encoded_ids.append(self.inverse_vocab[token.encode("utf-8")])
                continue
                
            word_bytes = token.encode("utf-8")
            encoded_ids.extend(self._encode_chunk(word_bytes))
        return encoded_ids

    # def encode(self, text: str) -> list[int]:
    #     tokens = pre_tokenize_string(text, self.special_tokens, keep_special_tokens=True)
    #     return self._encode_tokens(tokens)

    def _encode_single_text(self, text: str) -> list[int]:
        """
        Helper method to encode a single text string without multiprocessing.
        """
        tokens = pre_tokenize_string(text, self.special_tokens, keep_special_tokens=True)
        return self._encode_tokens(tokens)

    def encode(self, text: str, num_processes: int = 4) -> list[int]:
        """
        Encodes text in parallel using multiple processes.
        """
        tokens = pre_tokenize_string(text, self.special_tokens, keep_special_tokens=True)
        
        # If no tokens, return empty list
        if not tokens:
            return []
            
        # Split tokens into chunks, ensuring at least 1024 tokens per chunk
        base_chunk_size = len(tokens) // num_processes + (1 if len(tokens) % num_processes > 0 else 0)
        chunk_size = max(base_chunk_size, 1024)
        
        token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        
        # If only one chunk, avoid overhead
        if len(token_chunks) <= 1:
            return self._encode_tokens(tokens)

        with multiprocessing.Pool(processes=num_processes) as pool:
            # We need to bind the instance method to the instance for pickle
            # However, instance methods are picklable in Python 3.
            results = pool.map(self._encode_tokens, token_chunks)
            
        # Merge results
        final_ids = []
        for res in results:
            final_ids.extend(res)
            
        return final_ids
    
    def encode_iterable(self, iterable: Iterable[str], num_processes: int = 4) -> Iterator[int]:
        """
        Encodes an iterable of strings in parallel using a process pool.
        """
        with multiprocessing.Pool(processes=num_processes) as pool:
            for encoded_ids in pool.imap(self._encode_single_text, iterable, chunksize=100):
                yield from encoded_ids

    def decode(self, ids: list[int]) -> str:
        tokens = b"".join([self.vocab[id] for id in ids])
        return tokens.decode("utf-8", errors="replace")