import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=500):
        super().__init__()
        
        # Create positional encodings
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Project inputs
        Q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size)
        K = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size)
        V = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, heads, seq_len, head_size]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_size)
        
        # Apply mask (only for non-padding keys)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf')
            )
        
        # Softmax and attention
        attention_probs = torch.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.output_proj(context)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x, mask=None):
        # Attention with residual connection
        attn_output = self.attention(self.layer_norm1(x), mask)
        x = x + attn_output
        
        # MLP with residual connection
        mlp_output = self.mlp(self.layer_norm2(x))
        x = x + mlp_output
        
        return x

class ANBNTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=56, num_heads=2, max_len=202):
        super().__init__()
        
        # Embedding matrices
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, max_len)
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) 
            for _ in range(3)
        ])
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, 2)
    
    def forward(self, x, mask=None):
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.final_layer_norm(x)
        
        # Classification (using start token)
        logits = self.classifier(x[:, 0, :])
        return logits

def tokenize(string, max_len=202):
    """
    Tokenize a string according to the specified mapping
    1: start symbol
    3: 'a'
    4: 'b'
    2: end symbol
    0: padding
    """
    base = [1] + [3 if char == 'a' else 4 for char in string] + [2]
    return base + [0] * (max_len - len(base))

def generate_anbn_dataset(is_train=True):
    """
    Generate A^nB^n dataset
    
    For training:
    - 100 positive samples from length 1 to 100
    - 100 negative samples of even length less than 100
    
    For testing:
    - 100 positive samples from length 100 to 200
    - 100 negative samples from length 100 to 200
    """
    X, y = [], []
    
    if is_train:
        # Positive samples (1 to 100 length)
        for n in range(1, 101):
            string = "a" * n + "b" * n
            X.append(tokenize(string))
            y.append(1)
        
        # Negative samples (even length less than 100)
        while len(y) < 200:
            # Generate random even length strings less than 100
            length = np.random.randint(2, 100) 
            if length % 2 == 1:
                length += 1
            
            # Ensure the string is not a valid A^nB^n
            while True:
                # Generate a random string of the specified length
                string = ''.join(np.random.choice(['a', 'b'], length))
                
                # Check it's not a valid A^nB^n
                a_count = string.count('a')
                b_count = string.count('b')
                if a_count != b_count or a_count == 0:
                    X.append(tokenize(string))
                    y.append(0)
                    break
    else:
        # Positive samples (length 100 to 200)
        for n in range(100, 201):
            if n % 2 == 0:
                string = "a" * (n//2) + "b" * (n//2)
                X.append(tokenize(string))
                y.append(1)
        
        # Negative samples (length 100 to 200)
        while len(y) < 200:
            # Generate random even length strings between 100 and 200
            length = np.random.randint(100, 201)
            if length % 2 == 1:
                length += 1
            
            # Ensure the string is not a valid A^nB^n
            while True:
                string = ''.join(np.random.choice(['a', 'b'], length))
                
                # Check it's not a valid A^nB^n
                a_count = string.count('a')
                b_count = string.count('b')
                if a_count != b_count or a_count == 0:
                    X.append(tokenize(string))
                    y.append(0)
                    break
    
    return torch.tensor(X), torch.tensor(y)
