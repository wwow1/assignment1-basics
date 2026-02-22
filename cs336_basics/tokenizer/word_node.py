from __future__ import annotations

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
    self.prev = None
    self.next = None

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