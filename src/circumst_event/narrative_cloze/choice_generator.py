"""
Author       : Shichao Wang
Date         : 2021-01-25 20:24:58
LastEditTime : 2021-03-15 20:45:41
"""


import copy
import json
import os
import random
from abc import ABCMeta, abstractmethod
from itertools import chain
from operator import itemgetter
from typing import List, Sequence

from circumst_event.preprocessing.models import IndexedChain, IndexedEvent

FIELDS_MAP = {
    "subject": ("subject", "subject_ids"),
    "object": ("object", "object_ids"),
    "preposition": ("preposition", "preposition_ids"),
}


class AbstractChoiceGenerator(metaclass=ABCMeta):
    @abstractmethod
    def sample_a_event(self) -> IndexedEvent:
        raise NotImplementedError()

    def generate(self, event: IndexedEvent) -> IndexedEvent:
        """
        Generate one distracter given by target event
        :param event:
        :return: distacter event
        """
        ref_event = self.sample_a_event()
        new_event = copy.deepcopy(ref_event)

        be_replace_fields = random.choice(list(FIELDS_MAP))
        to_replace_fields = random.choice(list(FIELDS_MAP))
        for b, t in zip(
            FIELDS_MAP[be_replace_fields],
            FIELDS_MAP[to_replace_fields],
        ):
            # noinspection PyTypedDict
            new_event[b] = event[t]
        return new_event

    def generate_choices(
        self, event: IndexedEvent, num_choices: int
    ) -> List[IndexedEvent]:
        choices = [event]
        while len(choices) < num_choices:
            new_event = self.generate(event)
            if new_event not in choices:
                choices.append(new_event)
        return choices


class EventPoolChoiceGenerator(AbstractChoiceGenerator):
    def __init__(self, event_pool: Sequence[IndexedEvent]) -> None:
        self._event_pool = event_pool

    @classmethod
    def from_chains(cls, chains: List[IndexedChain]):
        events = list(chain.from_iterable(map(itemgetter("events"), chains)))
        return cls(events)

    def sample_a_event(self) -> IndexedEvent:
        return random.choice(self._event_pool)


class LazyChoiceGenerator(AbstractChoiceGenerator):
    def __init__(self, data_folder_path: str):
        self._data_folder_path = data_folder_path

    def sample_a_event(self) -> IndexedEvent:
        example_file = random.choice(os.listdir(self._data_folder_path))
        event_chain: IndexedChain = json.load(
            open(os.path.join(self._data_folder_path, example_file))
        )
        return random.choice(event_chain["events"])
