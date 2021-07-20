import itertools
from abc import abstractmethod
from typing import List, Mapping

from preprocessing.models import Event, EventChain, IndexedChain, IndexedEvent


def pad_and_truncate_to_length(
    tokens: List[str],
    pad_token: str,
    length: int,
) -> List[str]:
    padded = tokens + [pad_token] * length
    return padded[:length]


class AbstractEventIndexer:
    @property
    @abstractmethod
    def missing_token(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def padding_token(self):
        raise NotImplementedError()

    def index_argument(self, argument: List[str], argument_length: int):
        raise NotImplementedError()

    def index_sentence(self, sentence: List[str], max_sentence_length: int):
        raise NotImplementedError()

    def index_event(self, event: Event, argument_length: int) -> IndexedEvent:
        predicate_tokens = self.index_argument(event["predicate"], argument_length)
        subject_tokens = self.index_argument(
            event["subject"] or [self.missing_token], argument_length
        )
        object_tokens = self.index_argument(
            event["object"] or [self.missing_token], argument_length
        )
        preposition_tokens = self.index_argument(
            event["preposition"] or [self.missing_token], argument_length
        )
        return IndexedEvent(
            predicate=event["predicate"],
            predicate_ids=predicate_tokens,
            subject=event["subject"],
            subject_ids=subject_tokens,
            object=event["object"],
            object_ids=object_tokens,
            preposition=event["preposition"],
            preposition_ids=preposition_tokens,
        )

    def index_chain(
        self,
        event_chain: EventChain,
        argument_length: int,
        max_sequence_length: int,
    ):
        return IndexedChain(
            document_id=event_chain["document_id"],
            events=[
                self.index_event(event, argument_length)
                for event in event_chain["events"]
            ],
            event_sentence_indexes=event_chain["event_sentence_indexes"],
            sentences=event_chain["sentences"],
            sentences_ids=[
                self.index_sentence(sentence, max_sequence_length)
                for sentence in event_chain["sentences"]
            ],
        )


# class BertEventIndexer(AbstractEventIndexer):
#     def __init__(self, pretrained_model_path: str) -> None:
#         self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
#
#     def index_argument(
#         self, tokens: List[str], argument_length: int
#     ) -> List[int]:
#         return self._tokenizer(
#             tokens,
#             add_special_tokens=False,
#             max_length=argument_length,
#             padding="max_length",
#             is_split_into_words=True,
#             truncation=True,
#         )["input_ids"]
#
#     def index_sentence(self, sentence: List[str], max_sentence_length: int):
#         return self._tokenizer(
#             sentence,
#             padding="max_length",
#             is_split_into_words=True,
#             max_length=max_sentence_length,
#             truncation=True,
#         )["input_ids"]
#
#     @property
#     def padding_token(self):
#         return self._tokenizer.pad_token
#
#     @property
#     def missing_token(self):
#         return self._tokenizer.unk_token
#


class VocabEventIndexer(AbstractEventIndexer):
    def __init__(
        self,
        words: List[str],
        special_tokens: Mapping[str, int],
        missing_token: str,
        padding_token: str,
    ):
        self._missing_token = missing_token
        self._padding_token = padding_token

        if missing_token not in words:
            words.insert(0, missing_token)
        if padding_token not in words:
            words.insert(0, padding_token)

        special_ids = set(special_tokens.values())
        counter = filter(lambda i: i not in special_ids, itertools.count())
        self._word2id = dict(special_tokens)
        assert len(self._word2id) == len(special_ids)  # each special has unique id

        for sp in special_tokens:
            words.remove(sp)

        for word in words:
            index = next(counter)
            self._word2id[word] = index

    @property
    def word2id(self):
        return self._word2id

    @property
    def missing_token(self):
        return self._missing_token

    @property
    def missing_id(self):
        return self._word2id[self.missing_token]

    @property
    def padding_token(self):
        return self._padding_token

    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self._word2id.get(token, self.missing_id) for token in tokens]

    def index_argument(self, argument: List[str], argument_length: int):
        fitted_argument = pad_and_truncate_to_length(
            argument, self.padding_token, argument_length
        )
        return self._tokens_to_ids(fitted_argument)

    def index_sentence(self, sentence: List[str], max_sentence_length: int):
        fitted_sentence = pad_and_truncate_to_length(
            sentence, self.padding_token, max_sentence_length
        )
        return self._tokens_to_ids(fitted_sentence)
