from csv import reader
from jamo import h2j, j2hcj
from os.path import exists
from datetime import timedelta
from timeit import default_timer
from torch import torch, nn, optim
from torch.utils.data import DataLoader, TensorDataset

start = default_timer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not exists("out/data.csv"):
    print("Please generate the dataset with \"data.py\"")
    exit()

def getElapsed():
    elapsed = timedelta(seconds=default_timer() - start)
    return str(elapsed).split(".")[0]

with open("out/data.csv", encoding="utf-8", newline="") as file:
    data = [tuple(row) for row in reader(file)][:10]
    print(f"Loaded {len(data)} pairs in {getElapsed()}\n")
    romaja, korean = zip(*data)

class Initialize:
    def __init__(self, words):
        self.words = words
        self.charset = sorted(list(set("".join(words))))
        self.max = max([len(word) for word in words])
        self.tensors = []

def decompose(word):
    return ".".join([j2hcj(h2j(syllable)) for syllable in word])

romaja, korean = map(Initialize, [romaja, [decompose(word) for word in korean]])
charset_max = max(len(romaja.charset), len(korean.charset))

def create_tensors(input: Initialize):
    for word in input.words:
        tensor = torch.zeros(max(romaja.max, korean.max), dtype=torch.long)
        for i, char in enumerate(word):
            tensor[i] = input.charset.index(char)
        input.tensors.append(tensor)

create_tensors(romaja), create_tensors(korean)

class LSTM(nn.Module):
    def __init__(self, device):
        super(LSTM, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(0.3)
        self.embedding = nn.Embedding(charset_max, 256)
        self.lstm = nn.LSTM(256, 512, num_layers=3, batch_first=True, bidirectional=True, dropout=0.3)
        self.linear = nn.Linear(1024, charset_max)
    
    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        hidden = (  
            torch.zeros(6, input.size(0), 512).to(self.device),
            torch.zeros(6, input.size(0), 512).to(self.device)
        )
        output, hidden = self.lstm(embedded, hidden)
        return self.linear(output), hidden
    
model = LSTM(device).to(device)

dataset = TensorDataset(torch.stack(romaja.tensors), torch.stack(korean.tensors))
t, v = torch.utils.data.random_split(dataset, [
    int(len(dataset) * 0.8),
    len(dataset) - int(len(dataset) * 0.8)
])

training = DataLoader(t, batch_size=128, shuffle=True)
validation = DataLoader(v, batch_size=128, shuffle=False)
    
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=5)

for epoch in range(50):
    model.train()
    training_loss = 0
    for r, k in training:
        r, k = r.to(device), k.to(device)
        optimizer.zero_grad()
        output, _ = model(r)
        k = k.view(-1)
        loss = criterion(output.view(-1, charset_max), k)
        training_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    avg_training_loss = training_loss / len(training)

    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for r, k in validation:
            r, k = r.to(device), k.to(device)
            output, _ = model(r)
            k = k.view(-1)
            loss = criterion(output.view(-1, charset_max), k)
            validation_loss += loss.item()
    
    avg_validation_loss = validation_loss / len(validation)
    scheduler.step(avg_validation_loss)
    
    print(f"[{getElapsed()}] Epoch {epoch + 1:02} of 50, Training Loss: {avg_training_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}")

torch.jit.script(model).save("out/model.pt")

print(f"\nFinished in {getElapsed()}")