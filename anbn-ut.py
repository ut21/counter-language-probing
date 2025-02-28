import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import math

torch.manual_seed(42)
random.seed(42)

def tokenize(string, max_len=202):
    base = [1] + [3 if char == 'a' else 4 for char in string] + [2]
    return base + [0] * (max_len - len(base))

# Dataset generation
def generate_positive_samples(n_min, n_max):
    n_max = n_max if n_max % 2 == 0 else n_max - 1
    n_max = int(n_max/2)
    n_min = int(n_min/2)
    return [("a" * n + "b" * n, 1) for n in range(n_min, n_max + 1)]

def generate_negative_samples(length_min, length_max, n_samples):
    samples = []
    while len(samples) < n_samples:
        length = random.randint(length_min, length_max)
        string = "".join(random.choices("ab", k=length))
        mid = string.find("b")
        if mid != -1 and string[:mid] == "a" * mid and string[mid:] == "b" * (length - mid):
            continue
        samples.append((string, 0))
    return samples

# Dataset and DataLoader
class StringDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        string, label = self.data[idx]
        return torch.tensor(tokenize(string)), label

positive_samples = generate_positive_samples(1, 100)  # Positive samples up to n=50
negative_samples = generate_negative_samples(2, 100, len(positive_samples))  # Equal number of negative samples
dataset = positive_samples + negative_samples

train_loader = DataLoader(StringDataset(dataset), batch_size=16, shuffle=True)
train_loader_positive = DataLoader(StringDataset(positive_samples), batch_size=16, shuffle=True)

positive_test_samples = generate_positive_samples(20, 200)
negative_test_samples = generate_negative_samples(20, 200, len(positive_test_samples))
print(len(positive_test_samples), len(negative_test_samples))
test_dataset = positive_test_samples + negative_test_samples

test_loader = DataLoader(StringDataset(test_dataset), batch_size=16, shuffle=False)
test_loader_positive = DataLoader(StringDataset(positive_test_samples), batch_size=16, shuffle=False)
test_loader_negative = DataLoader(StringDataset(negative_test_samples), batch_size=16, shuffle=False)
print("Train samples:", len(dataset))
print("Test samples:", len(test_dataset))

import torch
import torch.nn as nn
import math

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=5, embed_dim=64, num_heads=4, num_layers=1, hidden_dim=128, dropout=0.3):
        super(SmallTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Fully connected output layer
        self.fc = nn.Linear(embed_dim, 2)

    def generate_positional_encoding(self, seq_len, embed_dim, device):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * (-math.log(10000.0) / embed_dim))
        pos_enc = torch.zeros(seq_len, embed_dim, device=device)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)  # Shape: (1, seq_len, embed_dim)

    def forward(self, x, return_internal=False):
        device = x.device
        seq_len = x.size(1)
        embed_dim = self.embedding.embedding_dim

        # Embed input and add positional encoding
        x = self.embedding(x)  # Shape: (batch_size, seq_len, embed_dim)
        positional_encoding = self.generate_positional_encoding(seq_len, embed_dim, device)
        x = x + positional_encoding  # Shape: (batch_size, seq_len, embed_dim)

        # Transformer encoder expects input of shape (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # Shape: (seq_len, batch_size, embed_dim)
        internal_representation = self.transformer_encoder(x)  # Shape: (seq_len, batch_size, embed_dim)
        
        # pool the internal representation
        cls_rep = internal_representation.mean(dim=0)  # Shape: (batch_size, embed_dim)

        # Classification output
        output = self.fc(cls_rep)  # Shape: (batch_size, 2)

        if return_internal:
            return output, internal_representation.permute(1, 0, 2)  # Shape: (batch_size, seq_len, embed_dim)
        else:
            return output

device = 'mps'

def evaluate_model(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get the model output and internal representation
            outputs = model(inputs)  # We're only interested in the output for evaluation
            
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

loss_hist = []
def train_model(model, train_loader, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get the model output and internal representation
            outputs = model(inputs)  # We're only interested in the output for loss calculation
            
            # Calculate the loss
            loss = criterion(outputs, labels)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loss_hist.append(loss.item())
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        # scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / total:.2%}")
        
        if (epoch+1) % 25 == 0:
            acc = evaluate_model(model, test_loader)
            print(f"Test accuracy: {acc:.2%}")

probe_model = SmallTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(probe_model.parameters(), lr=3e-4, weight_decay=1e-5)

EPOCHS = 250
train_model(probe_model, train_loader, epochs = EPOCHS)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))

print(f"Test accuracy: {evaluate_model(probe_model, test_loader):.2%}")
print(f"Positive test accuracy: {evaluate_model(probe_model, test_loader_positive):.2%}")
print(f"Negative test accuracy: {evaluate_model(probe_model, test_loader_negative):.2%}")
print(f"Positive train accuracy: {evaluate_model(probe_model, train_loader_positive):.2%}")
print(f"Train accuracy: {evaluate_model(probe_model, train_loader):.2%}")

plt.plot(loss_hist)

def simulate_stack(sequence):
    stack = []
    labels = []
    flag = False
    for token in sequence:
        if token == 4:
            flag = True
        if token == 3 and not flag:
            stack.append(1)
        elif token == 4 and stack:
            stack.pop()
        labels.append(stack[-1] if stack else 0)
    return labels

def create_probing_dataset(model, data_loader, device):
    model.eval()
    probing_features = []
    probing_labels = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _, internal_representations = model(inputs, return_internal=True)  # Shape: (batch_size, seq_len, embed_dim)

            for batch_idx in range(inputs.size(0)):
                sequence = inputs[batch_idx].cpu().numpy()
                hidden_states = internal_representations[batch_idx]  # Shape: (seq_len, embed_dim)
                labels = simulate_stack(sequence)

                probing_features.append(hidden_states[:len(labels)])
                probing_labels.extend(labels)
                
    print(len(probing_features), len(probing_labels))
    probing_features = torch.cat(probing_features)  # Shape: (total_tokens, embed_dim)
    probing_labels = torch.tensor(probing_labels)  # Shape: (total_tokens,)
    
    return torch.utils.data.TensorDataset(probing_features, probing_labels)

def create_control_probing_dataset(model, data_loader, device):
    model.eval()
    probing_features = []
    probing_labels = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _, internal_representations = model(inputs, return_internal=True)

            for batch_idx in range(inputs.size(0)):
                sequence = inputs[batch_idx].cpu().numpy()
                hidden_states = internal_representations[batch_idx]
                labels = [random.choice([0, 1]) for _ in range(len(sequence))]
                probing_features.append(hidden_states)
                probing_labels.extend(labels)
    
    probing_features = torch.cat(probing_features)
    probing_labels = torch.tensor(probing_labels)
    
    return torch.utils.data.TensorDataset(probing_features, probing_labels)

create_control_probing_dataset(probe_model, train_loader, device)

class ProbingClassifier(nn.Module):
    def __init__(self, embed_dim, output_dim, linear=True):
        super(ProbingClassifier, self).__init__()
        if linear:
            self.fc = nn.Linear(embed_dim, output_dim)
        else:
            self.fc = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, output_dim)
            )

    def forward(self, x):
        return self.fc(x)

def train_probing_classifier(dl, probing_model, epochs=10):
    loss_probe = []
    accuracy_probe = []
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for features, labels in dl:
            features, labels = features.to(device), labels.to(device)
            probing_model.train()
            outputs = probing_model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_probe.append(loss.item())
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            accuracy_probe.append(correct / total)
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dl):.4f}, Accuracy: {correct / total:.2%}")

    return loss_probe, accuracy_probe

def evaluate_probing_classifier(probe, test_dl):
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_dl:
            features, labels = features.to(device), labels.to(device)
            outputs = probe(features)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    accuracy = correct / total
    print(f"Probing Classifier Accuracy: {accuracy:.2%}")
    return correct / total

batch_size = 64

train_ds_control = create_control_probing_dataset(probe_model, train_loader, device)
train_dl_control = DataLoader(train_ds_control, batch_size=batch_size, shuffle=True)

test_ds_control = create_control_probing_dataset(probe_model, test_loader, device)
test_dl_control = DataLoader(test_ds_control, batch_size=batch_size, shuffle=False)

embed_dim = 64
control_probe = ProbingClassifier(embed_dim, output_dim=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(control_probe.parameters(), lr=3e-4)

probe_loss_hist_control, probe_acc_hist_control = train_probing_classifier(train_dl_control, control_probe, epochs=30)
evaluate_probing_classifier(control_probe, test_dl_control)

batch_size = 64

train_ds_task = create_probing_dataset(probe_model, train_loader, device)
train_dl_task = DataLoader(train_ds_task, batch_size=batch_size, shuffle=True)

test_ds_task = create_probing_dataset(probe_model, test_loader, device)
test_dl_task = DataLoader(test_ds_task, batch_size=batch_size, shuffle=False)

embed_dim = 64
task_probe = ProbingClassifier(embed_dim, output_dim=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(task_probe.parameters(), lr=1e-3)

probe_loss_hist_task, probe_acc_hist_task = train_probing_classifier(train_dl_task, task_probe, epochs=30)
evaluate_probing_classifier(task_probe, test_dl_task)

untrained_model = SmallTransformer().to(device)

batch_size = 32

train_ds_random = create_probing_dataset(untrained_model, train_loader, device)
train_dl_random = DataLoader(train_ds_random, batch_size=batch_size, shuffle=True)

test_ds_random = create_probing_dataset(untrained_model, test_loader, device)
test_dl_random = DataLoader(test_ds_random, batch_size=batch_size, shuffle=False)

embed_dim = 64
untrained_task_probe = ProbingClassifier(embed_dim, output_dim=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(untrained_task_probe.parameters(), lr=1e-3)

untrained_losses_probe, untrained_accuracy_probe = train_probing_classifier(train_dl_random, untrained_task_probe, epochs=30)

print("Probing classifier results for untrained model:")
evaluate_probing_classifier(untrained_task_probe, test_dl_random)

plt.figure(figsize=(10, 5))

plt.plot(probe_loss_hist_task[::500], label="Trained model", color='black')
plt.plot(untrained_losses_probe[::500], label="Untrained model")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Classifier Loss")
plt.title("Probing for Top Element in Stack")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(probe_acc_hist_task[::100], label="Trained model", color='black')
plt.plot(untrained_accuracy_probe[::100], label="Untrained model")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy of classifier")
plt.title("Probing for Top Element in Stack")
plt.show()

print(len(probe_acc_hist_control))
len(probe_acc_hist_task)

plt.figure(figsize=(10, 5))
plt.plot(probe_acc_hist_task[::1000], label="Top Element of Stack", color='black')
plt.plot(probe_acc_hist_control[::500], label="Control task")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy of classifier")
plt.title("Classifier Accuracy for Stack Prediction vs Control Task")
plt.show()

class ProbingClassifier_ablation(nn.Module):
    def __init__(self, embed_dim, output_dim, n_layers):
        super(ProbingClassifier_ablation, self).__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.layers(x)
        return self.fc(x)

def train_probing_classifier_ablation(dl, embed_dim, n_layers, epochs=10, batch_size=64):
    # Instantiate the probing classifier with given `n_layers`
    model = ProbingClassifier_ablation(embed_dim=embed_dim, output_dim=2, n_layers=n_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_probe = []
    accuracy_probe = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        for features, labels in dl:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        loss_probe.append(total_loss / len(dl))
        accuracy_probe.append(correct / total)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dl):.4f}, Accuracy: {correct / total:.2%}")

    return model, loss_probe, accuracy_probe

n_layers = [1, 5, 6, 7, 10]
stack_accuracies = []
control_accuracies = []
selectivities = []

for n_layer in n_layers:
    print(f"\nTraining Probing Classifier with Number of Layers: {n_layer}")

    # Create probing datasets for stack and control tasks
    # task_ds = create_probing_dataset(probe_model, train_loader, device)
    # task_dl = DataLoader(task_ds, batch_size=batch_size, shuffle=True)
    
    control_ds = create_control_probing_dataset(probe_model, train_loader, device)
    control_dl = DataLoader(control_ds, batch_size=batch_size, shuffle=True)

    # Train probing classifier for stack task
    # stack_classifier, _,  stack_accuracy = train_probing_classifier_ablation(
    #     task_dl, embed_dim=64, n_layers=n_layer, epochs=30
    # )
    
    # evaluate_probing_classifier(stack_classifier, task_dl)
    # stack_accuracies.append(stack_accuracy[-1])

    # Train probing classifier for control task
    control_classifier, _, control_accuracy = train_probing_classifier_ablation(
        control_dl, embed_dim=64, n_layers=n_layer, epochs=30
    )
    evaluate_probing_classifier(control_classifier, control_dl)
    control_accuracies.append(control_accuracy[-1])

    # Calculate selectivity
    # for i in range(len(stack_accuracies)):
        # selectivities.append(stack_accuracies[i] - control_accuracies[i])


plt.figure(figsize=(10, 6))
plt.plot(n_layers, stack_accuracies, label="Stack Task Accuracy", marker="o", color="skyblue")
plt.plot(n_layers, control_accuracies, label="Control Task Accuracy", marker="o", color="salmon")
# plt.bar([str(layer) for layer in n_layers], selectivities, alpha=0.3, label="Selectivity", color="blue", width=0.4)

plt.xlabel("Number of Layers in Probing Classifier")
plt.ylabel("Accuracy")
plt.title("Probe Accuracy and Selectivity Across Number of Layers")
plt.legend()
plt.grid()
plt.show()