"""One encoder, five heads: domain, field presence, BIO span tags, span operator, span-less (implied) values."""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from .data import DOMAINS, OPS, Vocab


class TaggerModel(nn.Module):
    def __init__(self, encoder_name: str, vocab: Vocab, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, add_pooling_layer=False)
        h = self.encoder.config.hidden_size
        self.vocab = vocab
        self.drop = nn.Dropout(dropout)
        self.domain = nn.Linear(h, len(DOMAINS))
        self.present = nn.Linear(h, len(vocab.fields))
        self.tags = nn.Linear(h, len(vocab.tags))
        self.op = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, len(OPS)))
        self.implied_fields = list(vocab.implied)
        self.implied = nn.ModuleList(nn.Linear(h, len(vocab.implied[f])) for f in self.implied_fields)
        # categorical value head over all (field, value) pairs; logits are masked to the span's field
        n_val = len(vocab.value_index)
        self.value = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, n_val)) if n_val else None
        mask = torch.full((len(vocab.fields), max(1, n_val)), float("-inf"))
        fidx = {f: i for i, f in enumerate(vocab.fields)}
        for j, (f, _) in enumerate(vocab.value_index):
            mask[fidx[f], j] = 0
        self.register_buffer("value_mask", mask, persistent=False)
        self._fidx = fidx

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def forward(self, input_ids, attention_mask):
        hid = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hid = self.drop(hid)
        cls = hid[:, 0]
        return {"hidden": hid, "domain": self.domain(cls), "present": self.present(cls), "tags": self.tags(hid),
                "implied": [m(cls) for m in self.implied]}

    def span_logits(self, hidden, spans):
        """spans: list of (batch_index, tok_start, tok_end). Span = [mean of its tokens ; its first token].
        Returns (operator logits, value logits or None)."""
        if not spans:
            return hidden.new_zeros((0, len(OPS))), None
        reps = torch.stack([torch.cat([hidden[b, s:e].mean(0), hidden[b, s]]) for b, s, e in spans])
        return self.op(reps), (self.value(reps) if self.value is not None else None)

    def value_mask_for(self, fields: list[str]):
        return self.value_mask[[self._fidx[f] for f in fields]]

    def op_logits(self, hidden, spans):
        return self.span_logits(hidden, spans)[0]
