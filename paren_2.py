import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from typing import Union
from jaxtyping import Int
from torch import Tensor
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

VOCAB = {"[pad]": 0, "[start]": 1, "[end]": 2, "(": 3, ")": 4}
HIDDEN_SIZE = 56
HEAD_SIZE = 28
NUM_LAYERS = 3
NUM_HEADS = 2
MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
device = "mps"


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


class BracketsDataset(torch.utils.data.Dataset):
    def __init__(self, data_tuples, tokenizer):
        self.tokenizer = SimpleTokenizer("()")
        self.strs = [x[0] for x in data_tuples]
        self.isbal = torch.tensor([x[1] for x in data_tuples])
        self.toks = self.tokenizer.tokenize(self.strs)
        self.open_proportion = torch.tensor([s.count("(") / len(s) for s in self.strs])
        self.starts_open = torch.tensor([s[0] == "(" for s in self.strs]).bool()

    def __len__(self):
        return len(self.strs)

    def __getitem__(self, idx):
        return self.strs[idx], self.isbal[idx], self.toks[idx]

    def to(self, device):
        self.isbal = self.isbal.to(device)
        self.toks = self.toks.to(device)
        self.open_proportion = self.open_proportion.to(device)
        self.starts_open = self.starts_open.to(device)
        return self

def load_data():
    with open("/Users/utkarsh/Documents/neural-toc/naacl_work/ARENA_3.0/chapter1_transformer_interp/exercises/part51_balanced_bracket_classifier/brackets_data.json") as f:
        data_tuples = json.load(f)
    data_tuples = data_tuples
    random.shuffle(data_tuples)

    train_size = int(0.8 * len(data_tuples))
    val_size = int(0.1 * len(data_tuples))
    test_size = len(data_tuples) - train_size - val_size

    train_data = data_tuples[:train_size]
    val_data = data_tuples[train_size:train_size+val_size]
    test_data = data_tuples[train_size+val_size:]

    tokenizer = SimpleTokenizer("()")
    train_dataset = BracketsDataset(train_data, tokenizer).to(device)
    val_dataset = BracketsDataset(val_data, tokenizer).to(device)
    test_dataset = BracketsDataset(test_data, tokenizer).to(device)

    return train_dataset, val_dataset, test_dataset

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.W_q = nn.Linear(hidden_size, num_heads * head_size)
        self.W_k = nn.Linear(hidden_size, num_heads * head_size)
        self.W_v = nn.Linear(hidden_size, num_heads * head_size)
        self.W_o = nn.Linear(num_heads * head_size, hidden_size)

    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_size = x.size()
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, -1)
        return self.W_o(context)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, head_size, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, head_size, num_heads)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.layernorm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.layernorm1(x + attn_output)
        mlp_output = self.mlp(x)
        return self.layernorm2(x + mlp_output)

# class BalancedParenthesesModel(nn.Module):
#     def __init__(self, vocab_size, hidden_size, max_len, num_layers, num_heads):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, hidden_size)
#         self.positional_encodings = nn.Parameter(torch.zeros(1, max_len, hidden_size))
#         self.layers = nn.ModuleList([
#             TransformerBlock(hidden_size, HEAD_SIZE, num_heads)
#             for _ in range(num_layers)
#         ])
#         self.layernorm_final = nn.LayerNorm(hidden_size)
#         self.unembedding = nn.Linear(hidden_size, 2)

#     def forward(self, x, mask=None):
#         seq_len = x.size(1)
#         x = self.embedding(x) + self.positional_encodings[:, :seq_len, :]
#         for layer in self.layers:
#             x = layer(x, mask)
#         x = self.layernorm_final(x)
#         logits = self.unembedding(x[:, 0, :])
#         return logits

class BalancedParenthesesModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len, num_layers, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encodings = nn.Parameter(torch.zeros(1, max_len, hidden_size))
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, HEAD_SIZE, num_heads)
            for _ in range(num_layers)
        ])
        self.layernorm_final = nn.LayerNorm(hidden_size)
        self.unembedding = nn.Linear(hidden_size, 2)
        
        # Store intermediate states during forward pass
        self.token_states = None

    def forward(self, x, mask=None, return_states=False):
        seq_len = x.size(1)
        
        # Embedding with positional encodings
        x = self.embedding(x) + self.positional_encodings[:, :seq_len, :]
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final layer normalization
        x = self.layernorm_final(x)
        
        # Compute logits
        logits = self.unembedding(x[:, 0, :])
        
        # Store token states if requested
        if return_states:
            return logits, x.detach()
        
        return logits

