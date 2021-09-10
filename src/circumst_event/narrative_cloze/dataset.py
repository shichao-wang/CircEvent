"""
Authors:    Shichao Wang
Created at: 2021-07-20 15:09:08
"""
import gzip
import json
import os
import random
from typing import Collection, Iterator, List, Optional, Union

import pytorch_lightning as pl
import torch
from more_itertools import islice_extended
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from circumst_event.narrative_cloze.models import (
    MCNCWithSentence,
    MCNCWithSentenceBatch,
)
from circumst_event.preprocessing.models import IndexedChain, IndexedEvent


def get_events_tensor(
    indexed_events: Collection[IndexedEvent],
    argument_length: int,
    num_to_pad: int,
):
    def _event_as_tensor(indexed_event: IndexedEvent):
        predicate_ids = indexed_event["predicate_ids"]
        assert len(predicate_ids) == argument_length

        subject_ids = indexed_event["subject_ids"]
        assert len(subject_ids) == argument_length

        object_ids = indexed_event["object_ids"]
        assert len(object_ids) == argument_length

        return torch.as_tensor(
            [predicate_ids, subject_ids, object_ids], dtype=torch.long
        )

    event_tensors = [_event_as_tensor(event) for event in indexed_events]
    return torch.stack(
        event_tensors
        + [torch.zeros_like(event_tensors[0], dtype=torch.long)] * num_to_pad,
    )


class MCNCDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        num_chains: int = -1,
        num_choices: int = 5,
        argument_length: int = 15,
        context_events_length: int = 8,
        *,
        small: bool = False,
    ):
        super(MCNCDataset, self).__init__()
        self._data_file = data_file
        chain_iter: Iterator[IndexedChain] = map(
            json.loads,
            tqdm(gzip.open(data_file, "rt"), desc="Loading Examples"),
        )
        chain_iter = islice_extended(chain_iter, num_chains)
        self._examples = [
            chain
            for chain in chain_iter
            if len(chain["events"]) >= context_events_length
        ]
        print(f"Here are {len(self._examples)} examples.")
        self._num_choices = num_choices
        self._argument_length = argument_length
        self._context_events_length = context_events_length
        self._choice_generator = EventPoolChoiceGenerator.from_chains(self._examples)

    def __getitem__(self, item: int) -> MCNCWithSentence:
        indexed_chain: IndexedChain = self._examples[item]
        events = indexed_chain["events"][: self._context_events_length + 1]
        # for event in events:
        #     event["document_id"] = indexed_chain["document_id"]
        context_events: List[IndexedEvent]
        [*context_events, target_event] = events
        num_to_pad_event = self._context_events_length - len(context_events)
        events_mask_tensor = torch.as_tensor(
            [True] * len(context_events) + [False] * num_to_pad_event,
            dtype=torch.bool,
        )

        choices = self._choice_generator.generate_choices(
            target_event, num_choices=self._num_choices
        )
        random.shuffle(choices)
        label = choices.index(target_event)
        events_tensor = get_events_tensor(
            context_events,
            argument_length=self._argument_length,
            num_to_pad=num_to_pad_event,
        )

        sentence_ids = torch.as_tensor(
            indexed_chain["sentences_ids"], dtype=torch.long
        )[indexed_chain["event_sentence_indexes"][: self._context_events_length]]
        sentences_ids_mask = torch.as_tensor(sentence_ids != 0, dtype=torch.long)
        event_sentence_indexes = torch.as_tensor(
            indexed_chain["event_sentence_indexes"][: self._context_events_length]
            + [-1] * num_to_pad_event,
            dtype=torch.long,
        )
        #
        event_sent_mat = torch.zeros(
            self._context_events_length,
            self._context_events_length,
            dtype=torch.float,
        )
        diags = [0] * self._context_events_length
        for sent_id in event_sentence_indexes:
            diags[sent_id] += 1
        count = 0
        for c in diags:
            event_sent_mat[count : count + c, count : count + c] = 1
            count += c

        event_sent_mat = event_sent_mat / torch.sum(event_sent_mat, dim=0)
        # for event_id, sent_id in enumerate(event_sentence_indexes):
        #     if sent_id == -1:
        #         continue
        #     event_sent_mat[event_id][sent_id] = 1
        #     event_sent_mat[sent_id][event_id] = 1

        return MCNCWithSentence(
            events_tensor=events_tensor,
            events_mask_tensor=events_mask_tensor,
            choices_tensor=get_events_tensor(
                choices, argument_length=self._argument_length, num_to_pad=0
            ),
            label_tensor=torch.as_tensor(label, dtype=torch.long),
            sentences_tensor=sentence_ids,
            sentences_ids_mask=sentences_ids_mask,
            event_sent_mat=event_sent_mat,
            # events=context_events,
            # choices=choices,
            # sentences=[
            #     indexed_chain["sentences"][sent_index]
            #     for sent_index in indexed_chain["event_sentence_indexes"][
            #         : self._context_events_length
            #     ]
            # ],
        )

    @staticmethod
    def collate_fn(item_list: List[MCNCWithSentence]):
        return MCNCWithSentenceBatch(
            events_tensor=torch.stack([mcnc.events_tensor for mcnc in item_list]),
            events_mask_tensor=torch.stack(
                [mcnc.events_mask_tensor for mcnc in item_list]
            ),
            choices_tensor=torch.stack([mcnc.choices_tensor for mcnc in item_list]),
            label_tensor=torch.stack([mcnc.label_tensor for mcnc in item_list]),
            sentences_tensor=torch.stack([mcnc.sentences_tensor for mcnc in item_list]),
            sentences_ids_mask=torch.stack(
                [mcnc.sentences_ids_mask for mcnc in item_list]
            ),
            event_sent_mat=torch.stack([mcnc.event_sent_mat for mcnc in item_list]),
            # events=[mcnc.events for mcnc in item_list],
            # choices=[mcnc.choices for mcnc in item_list],
            # sentences=[mcnc.sentences for mcnc in item_list],
        )

    def __len__(self):
        return len(self._examples)


class MCNCDataModule(pl.LightningDataModule):
    def prepare_data(self, *args, **kwargs):
        pass

    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        num_choices: int,
        argument_length: int,
        context_events_length: int,
        num_workers: int,
        pin_memory: bool = True,
        small: bool = False,
    ):
        super().__init__()
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self._num_choices = num_choices
        self._argument_length = argument_length
        self._context_events_length = context_events_length
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._small = small

        self._train = None
        self._dev = None
        self._test = None

    def _new_dataset(self, data_file: str):
        return MCNCDataset(
            os.path.join(self._dataset_path, data_file),
            num_chains=-1 if not self._small else 1000,
            num_choices=self._num_choices,
            argument_length=self._argument_length,
            context_events_length=self._context_events_length,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "test" or stage is None:
            self._test = self._test or self._new_dataset("test.json.gz")

        if stage == "fit" or stage is None:
            self._train = self._new_dataset("train.json.gz")
            self._dev = self._new_dataset("dev.json.gz")
            self._test = self._test or self._new_dataset("test.json.gz")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            # collate_fn=MCNCDataset.collate_fn,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return [
            DataLoader(
                dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
                # collate_fn=MCNCDataset.collate_fn,
            )
            for dataset in [self._dev, self._test]
        ]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self._test,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            # collate_fn=MCNCDataset.collate_fn,
        )
