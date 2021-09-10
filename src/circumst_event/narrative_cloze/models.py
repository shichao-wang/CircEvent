"""
Authors:    Shichao Wang
Created at: 2021-07-20 15:08:18
"""
from typing import Any, List, NamedTuple, Optional

import numpy
import pytorch_lightning as pl
import torch
from torch import BoolTensor, LongTensor, Tensor, nn
from torch.nn import functional as f
from torch.optim import Adam
from torchmetrics import Accuracy, MetricCollection

from torch_utils import load_glove_into_dict, mh_attention_forward, rnn_forward


class MCNarrativeCloze(NamedTuple):
    events_tensor: Tensor  # (bs, event_length, pred+args, length)
    events_mask_tensor: BoolTensor  # (bs, event_length)

    choices_tensor: Tensor  # (bs, num_choices, pred+args, length)
    label_tensor: LongTensor  # (bs,)


class MCNCWithSentence(NamedTuple):
    events_tensor: Tensor  # (bs, event_length, pred+args, length)
    events_mask_tensor: Tensor  # (bs, event_length)
    choices_tensor: Tensor  # (bs, num_choices, pred+args, length)
    label_tensor: Tensor  # (bs,)

    sentences_tensor: Tensor  # (bs, num_events, seq_len)
    sentences_ids_mask: Tensor  # (bs, num_events, seq_len)
    event_sent_mat: Tensor  # (bs, num_events)

    # events: List[IndexedEvent]
    # choices: List[IndexedEvent]
    # sentences: List[List[str]]


class MCNCWithSentenceBatch(NamedTuple):
    events_tensor: Tensor  # (bs, event_length, pred+args, length)
    events_mask_tensor: Tensor  # (bs, event_length)
    choices_tensor: Tensor  # (bs, num_choices, pred+args, length)
    label_tensor: Tensor  # (bs,)

    sentences_tensor: Tensor  # (bs, num_events, seq_len)
    sentences_ids_mask: Tensor  # (bs, num_events, seq_len)
    # sentences_mask_tensor: Tensor  # (bs, num_events, seq_len)
    event_sent_mat: Tensor  # (bs, num_events)
    #
    # events: List[List[IndexedEvent]]
    # sentences: List[List[List[str]]]
    # choices: List[List[IndexedEvent]]


class MCNCOutput(NamedTuple):
    logits: Tensor
    event_sent_weights: Optional[Tensor]


class CircEventLightningModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_path: str,
        freeze_embedding: bool,
        circ_mode: str,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
        loss_reg: float,
        reg_fn: str,
    ):
        super().__init__()
        # embedding
        glove_dict = load_glove_into_dict(pretrained_path)
        weights = torch.from_numpy(numpy.row_stack(list(glove_dict.values())))
        self._embedding = nn.Embedding.from_pretrained(
            embeddings=torch.as_tensor(weights, dtype=torch.float),
            freeze=freeze_embedding,
        )
        # event embedding
        embed_size = self._embedding.embedding_dim
        self._composition = nn.Sequential(
            nn.Linear(2 * 3 * embed_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        # sentence embedding
        self._sent_rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
            batch_first=True,
        )
        self._circ_mode = circ_mode
        self._local_mh = nn.MultiheadAttention(hidden_size, num_heads=num_heads)
        if self._circ_mode == "global":
            self._global_mh = nn.MultiheadAttention(hidden_size, num_heads=num_heads)
        self._linear = nn.Linear(2 * hidden_size, hidden_size)
        self._transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
        )
        self._output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._loss_reg = torch.as_tensor(loss_reg)
        self._reg_fn = reg_fn
        self._train_metric = MetricCollection({"acc_train": Accuracy(top_k=1)})
        self._dev_metric = MetricCollection({"acc_val": Accuracy(top_k=1)})
        self._test_metric = MetricCollection({"acc_test": Accuracy(top_k=1)})

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

    def forward(self, mcnc: MCNCWithSentenceBatch):
        batch_size, num_events, sequence_length = mcnc.sentences_tensor.shape

        # events embedding
        all_events_tensor = torch.cat([mcnc.events_tensor, mcnc.choices_tensor], dim=1)
        arguments_embedding = self._embedding(all_events_tensor)
        avg_embedding = torch.mean(arguments_embedding, dim=-2)
        max_embedding = torch.max(arguments_embedding, dim=-2)[0]
        events_argument_embedding = torch.cat([max_embedding, avg_embedding], dim=-1)
        all_events_embedding = self._composition(
            torch.flatten(events_argument_embedding, 2, -1)
        )
        context_events_embedding, choice_events_embedding = torch.tensor_split(
            all_events_embedding, (-5,), dim=1
        )
        event_sent_weights = None
        # sentence_embedding
        flattened_sent_hidden_states, last_state = rnn_forward(
            self._sent_rnn,
            embeddings=torch.flatten(self._embedding(mcnc.sentences_tensor), 0, 1),
            preserve_length=True,
            mask=torch.flatten(mcnc.sentences_ids_mask, 0, 1),
        )  # (bs*num_event, seq_len, hid)
        # (bs, num_events, hid), (bs, num_events, seq_len, hid)
        (flattened_local_circ_embedding, local_weights,) = mh_attention_forward(
            self._local_mh,
            torch.unsqueeze(torch.flatten(context_events_embedding, 0, 1), dim=1),
            flattened_sent_hidden_states,
            flattened_sent_hidden_states,
            key_padding=~torch.flatten(mcnc.sentences_ids_mask, 0, 1).bool(),
            return_attention_weights=True,
        )  # (bs*num_event, n_head, hid)
        # (bs, num_events, hid)
        local_circ_embedding = torch.squeeze(
            flattened_local_circ_embedding, dim=1
        ).unflatten(0, (batch_size, num_events))
        if self._circ_mode == "global":
            (global_circ_embedding, event_sent_weights,) = mh_attention_forward(
                self._global_mh,
                context_events_embedding,
                local_circ_embedding,
                local_circ_embedding,
                return_attention_weights=True,
            )
            circ_embedding = global_circ_embedding
        else:
            circ_embedding = local_circ_embedding

        trans_input = self._linear(
            torch.cat(
                [
                    context_events_embedding,
                    circ_embedding
                    if self._circ_mode != "no"
                    else context_events_embedding,
                ],
                dim=-1,
            )
            # torch.cat(
            #     [
            #         circ_embedding,
            #         circ_embedding,
            #     ],
            #     dim=-1,
            # )
        )

        output = torch.transpose(
            self._transformer(
                src=torch.transpose(trans_input, 0, 1),
                tgt=torch.transpose(choice_events_embedding, 0, 1),
            ),
            0,
            1,
        )
        logits = torch.squeeze(self._output(output), dim=-1)
        return MCNCOutput(
            logits=logits,
            event_sent_weights=event_sent_weights,
        )

    def compute_loss(
        self,
        output_dict: MCNCOutput,
        input_dict: MCNCWithSentenceBatch,
    ) -> Tensor:
        loss = f.cross_entropy(output_dict.logits, input_dict.label_tensor)

        if self._circ_mode == "global":
            if self._reg_fn == "bce":
                attention_loss = f.binary_cross_entropy(
                    output_dict.event_sent_weights, input_dict.event_sent_mat
                )
            elif self._reg_fn == "l1":
                attention_loss = f.l1_loss(
                    output_dict.event_sent_weights, input_dict.event_sent_mat
                )
            elif self._reg_fn == "l2":
                attention_loss = f.mse_loss(
                    output_dict.event_sent_weights, input_dict.event_sent_mat
                )
            else:
                attention_loss = loss.new_tensor(0)

            loss = loss + self._loss_reg * attention_loss

        return loss

    def training_step(self, batch: MCNCWithSentenceBatch, *args, **kwargs):
        output: MCNCOutput = self(batch)
        loss = self.compute_loss(output, batch)
        self._train_metric(torch.softmax(output.logits, dim=-1), batch.label_tensor)
        self.log("loss", loss.item())
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log_dict(self._train_metric.compute())

    def validation_step(
        self,
        batch: MCNCWithSentenceBatch,
        batch_idx: int,
        dataloader_idx: int,
        *args,
        **kwargs,
    ):
        output: MCNCOutput = self(batch)
        if dataloader_idx == 0:
            metric = self._dev_metric
        elif dataloader_idx == 1:
            metric = self._test_metric
        else:
            raise ValueError()

        batch_metrics = metric(torch.softmax(output.logits, dim=-1), batch.label_tensor)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self._dev_metric.compute())
        self.log_dict(self._test_metric.compute())

    def test_step(self, batch: MCNCWithSentenceBatch, batch_id, *args, **kwargs):
        output: MCNCOutput = self(batch)
        self._test_metric(torch.softmax(output.logits, dim=-1), batch.label_tensor)
        lengths = torch.sum(batch.events_mask_tensor, dim=-1)
        preds = torch.argmax(output.logits, dim=-1) == batch.label_tensor
        self.write_prediction(str(batch_id), [lengths, preds])

    def test_epoch_end(self, outputs) -> None:
        self.log_dict(self._test_metric.compute())
        self.trainer.evaluation_loop.predictions.to_disk()
