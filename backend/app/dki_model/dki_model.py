# improved_dkt_plus.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# -------------------------------
# 1. Dummy Dataset
# -------------------------------
num_students = 100
num_questions = 20
seq_len = 10

def generate_dummy_data(num_students, num_questions, seq_len):
    data = []
    for _ in range(num_students):
        questions = np.random.randint(0, num_questions, size=seq_len)
        answers = np.random.randint(0, 2, size=seq_len)
        seq = list(zip(questions, answers))
        data.append(seq)
    return data

data = generate_dummy_data(num_students, num_questions, seq_len)

# -------------------------------
# 2. Dataset
# -------------------------------
class DKTDataset(Dataset):
    def __init__(self, sequences, num_questions):
        self.sequences = [seq for seq in sequences if len(seq) > 1]
        self.num_questions = num_questions

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x_seq, y_seq = [], []
        for i in range(len(seq)-1):
            q_id, ans = seq[i]
            x_seq.append(q_id + ans * self.num_questions)
            y_seq.append(seq[i+1][1])
        return torch.tensor(x_seq, dtype=torch.long), torch.tensor(y_seq, dtype=torch.float)

def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    x_batch = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=-1)
    y_batch = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=-1)
    return x_batch, y_batch

dataset = DKTDataset(data, num_questions)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# -------------------------------
# 3. Improved DKT Model
# -------------------------------
class DKTModel(nn.Module):
    def __init__(self, num_questions, embedding_dim=64, hidden_size=128, dropout=0.2):
        super().__init__()
        self.num_questions = num_questions
        self.embedding = nn.Embedding(num_questions * 2, embedding_dim)
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True,
                            bidirectional=True, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_questions)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        mask = (x != -1)
        x = x.clone()
        x[x == -1] = 0
        x_emb = self.dropout(self.layernorm(self.embedding(x)))

        lengths = mask.sum(dim=1).cpu()
        packed_emb = nn.utils.rnn.pack_padded_sequence(x_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_emb)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        out = self.fc(out)
        out = self.sigmoid(out)
        out = out * mask.unsqueeze(-1)
        return out, mask

# -------------------------------
# 4. Training
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DKTModel(num_questions).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.BCELoss(reduction='sum')

def masked_bce_loss(outputs, targets, mask):
    loss = criterion(outputs, targets)
    return loss / mask.sum()

epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs, mask = model(x_batch)
        y_onehot = nn.functional.one_hot(y_batch.long(), num_classes=num_questions).float()
        loss = masked_bce_loss(outputs, y_onehot, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step(total_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# After training loop in improved_dkt_plus.py
torch.save(model.state_dict(), "dkt_model.pt")

# -------------------------------
# 5. Utilities
# -------------------------------
def recommend_next_questions(student_sequence, top_k=3):
    model.eval()
    x_seq = [q_id + ans * num_questions for q_id, ans in student_sequence]
    x_tensor = torch.tensor([x_seq], dtype=torch.long).to(device)
    with torch.no_grad():
        probs, _ = model(x_tensor)
        last_probs = probs[0, -1]
    top_questions = torch.topk(1 - last_probs, top_k).indices.tolist()
    return top_questions

def student_mastery_report(student_sequence, mastery_threshold=0.7):
    """Returns mastery probabilities, weakest concepts, and recommendations."""
    model.eval()
    x_seq = [q_id + ans * num_questions for q_id, ans in student_sequence]
    x_tensor = torch.tensor([x_seq], dtype=torch.long).to(device)
    with torch.no_grad():
        probs, _ = model(x_tensor)
        mastery_probs = probs[0, -1].cpu().numpy()
    weakest = np.argsort(mastery_probs)[:3].tolist()
    recommendations = recommend_next_questions(student_sequence, top_k=3)
    return {
        "mastery_probs": mastery_probs,
        "weakest_concepts": weakest,
        "recommendations": recommendations
    }