"""
Authors:    Shichao Wang
Created at: 2021-07-19 17:10:12
"""
from typing import List, TypedDict


class Event(TypedDict):
    predicate: List[str]
    subject: List[str]
    object: List[str]
    preposition: List[str]


class IndexedEvent(Event):
    predicate_ids: List[int]
    subject_ids: List[int]
    object_ids: List[int]
    preposition_ids: List[int]


class EventChain(TypedDict):
    document_id: str
    events: List[Event]
    event_sentence_indexes: List[int]
    sentences: List[List[str]]


class IndexedChain(TypedDict):
    document_id: str
    events: List[IndexedEvent]
    event_sentence_indexes: List[int]
    sentences: List[List[str]]
    sentences_ids: List[List[int]]
