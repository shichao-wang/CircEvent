"""
Authors:    Shichao Wang
Created at: 2021-07-20 09:24:05
"""
import json
import os
from operator import itemgetter
from typing import Iterator, List

from nltk.corpus import wordnet

from multi_relational_script_learning.dnee.events.extract_events import (
    extract_multi_faceted_events,
)
from circumst_event.preprocessing.models import Event, EventChain


# load wordnet at import stage.
wordnet.ensure_loaded()


def clean_empty_token(tokens: List[str]):
    return [t for t in tokens if t]


def dnee_event_chain_iterator(
    ann_filepath: str,
) -> Iterator[EventChain]:
    example_id = os.path.splitext(os.path.basename(ann_filepath))[0]
    ann_dict = json.load(open(ann_filepath))

    sent_tokens = [
        [token["originalText"] for token in sentence["tokens"]]
        for sentence in ann_dict["sentences"]
    ]
    valid_sentence_ids = [
        sid for sid, tokens in enumerate(sent_tokens) if len(tokens) < 60
    ]
    events_dict = extract_multi_faceted_events(ann_dict, no_sentiment=True)
    for entity_events in events_dict.values():
        if len(entity_events) == 0:
            continue

        sent_id: int
        chain_sentence_ids = [
            sent_id
            for sent_id in map(itemgetter("sentidx"), entity_events)
            if sent_id in valid_sentence_ids
        ]
        events: List[Event] = []
        event_sent_ids: List[int] = []
        for dict_event in entity_events:
            sent_id = dict_event["sentidx"]
            if sent_id not in chain_sentence_ids:
                continue

            event = Event(
                predicate=clean_empty_token(str.split(dict_event["predicate"], "_")),
                subject=clean_empty_token(dict_event.get("arg0", [])),
                object=clean_empty_token(dict_event.get("arg1", [])),
                preposition=clean_empty_token(dict_event.get("arg2", [])),
            )
            if len(event["subject"]) != 0 and len(event["object"]) != 0:
                events.append(event)
                event_sent_ids.append(sent_id)

        yield EventChain(
            events=events,
            document_id=example_id,
            event_sentence_indexes=[
                chain_sentence_ids.index(event_sent_id)
                for event_sent_id in event_sent_ids
            ],
            sentences=[sent_tokens[sent_id] for sent_id in chain_sentence_ids],
        )


def dnee_event_chains(ann_file_path: str):
    return [c for c in dnee_event_chain_iterator(ann_file_path)]
