
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
VOCAB = {"[pad]": 0, "[start]": 1, "[end]": 2, "(": 3, ")": 4, "[": 5, "]": 6, "{": 7, "}": 8}
HIDDEN_SIZE = 56
HEAD_SIZE = 28
NUM_LAYERS = 3
NUM_HEADS = 2
MAX_LEN = 110
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
device = "mps"
class BracketsDataset(torch.utils.data.Dataset):
    def __init__(self, data_tuples, tokenizer):
        self.tokenizer = SimpleTokenizer("()[]{}")
        self.strs = [x[0] for x in data_tuples]
        self.isbal = torch.tensor([x[1] for x in data_tuples])
        self.toks = self.tokenizer.tokenize(self.strs)
        self.open_proportion = torch.tensor([(s.count("(")+s.count("[")+s.count("{")) / len(s) for s in self.strs])
        self.starts_open = torch.tensor([(s[0] == "(" or s[0] == "[" or s== "{") for s in self.strs]).bool()

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
class SimpleTokenizer:
    START_TOKEN = 1
    PAD_TOKEN = 0
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

def load_data():
    with open("./dataset.json") as f:
        data_tuples = json.load(f)
    data_tuples = data_tuples
    random.shuffle(data_tuples)

    train_size = int(0.7 * len(data_tuples))
    val_size = int(0.1 * len(data_tuples))
    test_size = len(data_tuples) - train_size - val_size

    train_data = data_tuples[:train_size]
    val_data = data_tuples[train_size:train_size+val_size]
    test_data = data_tuples[train_size+val_size:]

    tokenizer = SimpleTokenizer("()[]{}")
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
        
        print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attention_weights, V).transpose(1, 2).contiguous()
        print(f"Context shape: {context.shape}")
        
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

class BalancedParenthesesModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len, num_layers, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encodings = nn.Parameter(torch.zeros(1, max_len, hidden_size))
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, hidden_size // num_heads, num_heads)
            for _ in range(num_layers)
        ])
        self.layernorm_final = nn.LayerNorm(hidden_size)
        self.unembedding = nn.Linear(hidden_size, 2)
        
        # Store intermediate states during forward pass
        self.token_states = None

    def forward(self, x, mask=None, return_states=False):
        seq_len = x.size(1)
        positional_encodings = self.positional_encodings[:, :seq_len, :].to(x.device)
        x = self.embedding(x) + positional_encodings
        
        # Prepare the mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Add dimensions for heads and sequence length
            mask = mask.to(x.device)
        
        # Debugging: Check shapes before layers
        print(f"Input to transformer layers: {x.shape}")
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Debugging: Check shape after transformer layers
        print(f"Output of transformer layers: {x.shape}")
        
        # Final layer normalization
        x = self.layernorm_final(x)
        
        # Compute logits
        logits = self.unembedding(x[:, 0, :])
        
        # Debugging: Check logits shape
        print(f"Logits shape: {logits.shape}")
        
        # Store token states if requested
        if return_states:
            return logits, x.detach()
        
        return logits

def train_model(model, train_dataset, val_dataset, num_epochs, batch_size, lr, device):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    losses = []
    acc = []

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch, (_, labels, tokens) in enumerate(train_loader):
            optimizer.zero_grad()
            mask = (tokens != VOCAB["[pad]"]).unsqueeze(1).unsqueeze(2).to(device)
            output = model(tokens, mask)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train_loader))
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for _, (_, labels, tokens) in enumerate(val_loader):
                mask = (tokens != VOCAB["[pad]"]).unsqueeze(1).unsqueeze(2).to(device)
                output = model(tokens, mask)
                loss = criterion(output, labels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        avg_train_loss = total_loss / len(train_loader)
        print(f"Validation Loss: {avg_val_loss}, Train Loss: {avg_train_loss}")

    print("Training Complete.")
def evaluate_model(model, test_dataset, device):
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    correct = 0
    total = 0
    incorrect = []
    with torch.no_grad():
        for batch, (_, labels, tokens) in enumerate(test_loader):
            mask = (tokens != VOCAB["[pad]"]).unsqueeze(1).unsqueeze(2).to(device)
            output = model(tokens, mask)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            incorrect.extend((predicted != labels).nonzero())

    print(f"Accuracy: {100*correct / total}")
train_dataset, val_dataset, test_dataset = load_data()
model = BalancedParenthesesModel(len(VOCAB), HIDDEN_SIZE, MAX_LEN, NUM_LAYERS, NUM_HEADS)
train_model(model, train_dataset, val_dataset, EPOCHS, BATCH_SIZE, LEARNING_RATE, device)



