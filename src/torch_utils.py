"""
Authors:    Shichao Wang
Created at: 2021-07-20 15:06:08
"""
from typing import Dict, Optional, Tuple, Union

import numpy
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm


def load_glove_into_dict(glove_filepath: str) -> Dict[str, numpy.ndarray]:
    word_embeddings = {}
    for line in tqdm(open(glove_filepath), "Loading Glove"):
        word, *vector = line.split()
        vec = numpy.asfarray(vector)
        word_embeddings[word] = vec
    return word_embeddings


def rnn_forward(
    rnn: nn.RNNBase,
    embeddings: Tensor,
    mask: Optional[Tensor],
    preserve_length: bool = False,
):
    """
    :param rnn: nn.RNN
    :param embeddings: (bs, seq, emb)
    :param mask: (bs, seq)
    :param preserve_length
    :return: (bs, seq, hid), (bs, hid)
    """

    if mask is None:
        lengths = torch.sum(
            torch.sum(embeddings, dim=-1, dtype=torch.bool), dtype=torch.long
        )
    else:
        lengths = torch.sum(mask, dim=1, dtype=torch.long)
    lengths = lengths.to("cpu")

    packed = pack_padded_sequence(
        embeddings,
        lengths=lengths,
        batch_first=rnn.batch_first,
        enforce_sorted=False,
    )
    packed_hidden_states: PackedSequence
    packed_hidden_states, last_hidden = rnn(packed)
    if isinstance(rnn, nn.LSTM):
        last_hidden = last_hidden[0]
    total_length = embeddings.size(1) if preserve_length else None
    hidden_states, _ = pad_packed_sequence(
        packed_hidden_states,
        batch_first=rnn.batch_first,
        total_length=total_length,
    )
    return hidden_states, torch.transpose(last_hidden, 0, 1).reshape(
        lengths.size(0), rnn.num_layers, -1
    )


def mh_attention_forward(
    mh: nn.MultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding: Optional[Tensor] = None,
    *,
    return_attention_weights: bool = False
) -> Union[Tuple[Tensor, Tensor], Tensor]:
    x, weights = mh(
        query=torch.transpose(query, 0, 1),
        key=torch.transpose(key, 0, 1),
        value=torch.transpose(value, 0, 1),
        key_padding_mask=key_padding,
    )
    x = torch.transpose(x, 0, 1)
    if return_attention_weights:
        return x, weights
    else:
        return x
