import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from typing import Union
from jaxtyping import Int
from torch import Tensor

VOCAB = {"[pad]": 0, "[start]": 1, "[end]": 2, "(": 3, ")": 4}
HIDDEN_SIZE = 56
HEAD_SIZE = 28
NUM_LAYERS = 3
NUM_HEADS = 2
MAX_LEN = 50
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
device = "mps"

def load_data(file):
    N_SAMPLES = 5000
    with open(file) as f:
        data_tuples: list[tuple[str, bool]] = json.load(f)
        # print(f"loaded {len(data_tuples)} examples")
    assert isinstance(data_tuples, list)
    data_tuples = data_tuples[:N_SAMPLES]
    data = BracketsDataset(data_tuples).to(device)
    data_mini = BracketsDataset(data_tuples[:100]).to(device)
    return data, data_mini, data_tuples


class SimpleTokenizer:
    START_TOKEN = 0
    PAD_TOKEN = 1
    END_TOKEN = 2
    base_d = {"[start]": START_TOKEN, "[pad]": PAD_TOKEN, "[end]": END_TOKEN}

    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        # the 3 is because there are 3 special tokens (defined just above)
        self.t_to_i = {**{c: i + 3 for i, c in enumerate(alphabet)}, **self.base_d}
        self.i_to_t = {i: c for c, i in self.t_to_i.items()}

    def tokenize(self, strs: list[str], max_len=None) -> Int[Tensor, "batch seq"]:
        def c_to_int(c: str) -> int:
            if c in self.t_to_i:
                return self.t_to_i[c]
            else:
                raise ValueError(c)

        if isinstance(strs, str):
            strs = [strs]

        if max_len is None:
            max_len = max((max(len(s) for s in strs), 1))

        ints = [
            [self.START_TOKEN]
            + [c_to_int(c) for c in s]
            + [self.END_TOKEN]
            + [self.PAD_TOKEN] * (max_len - len(s))
            for s in strs
        ]
        return torch.tensor(ints)

    def decode(self, tokens) -> list[str]:
        assert tokens.ndim >= 2, "Need to have a batch dimension"

        def int_to_c(c: int) -> str:
            if c < len(self.i_to_t):
                return self.i_to_t[c]
            else:
                raise ValueError(c)

        return [
            "".join(
                int_to_c(i.item()) for i in seq[1:] if i != self.PAD_TOKEN and i != self.END_TOKEN
            )
            for seq in tokens
        ]

    def __repr__(self) -> str:
        return f"SimpleTokenizer({self.alphabet!r})"
