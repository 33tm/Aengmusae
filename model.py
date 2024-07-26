from csv import reader
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

with open("out/data.csv", encoding="utf-8", newline="") as file:
    data = [tuple(row) for row in reader(file)]
    print(f"Loaded {len(data)} pairs\n")
    romaja, korean = zip(*data)

class Initialize:
    def __init__(self, words):
        self.words = words
        self.charset = list(set("".join(words)))
        self.max = max([len(word) for word in words])
        self.tensors = []

def decompose(word):
    output = ""
    for syllable in word:
        if (u := ord(syllable) - 0xAC00) < 0:
            raise Exception(f"Non-Korean syllable \"{syllable}\" in word \"{word}\"")
        l = chr(u // 588 + 0x1100)
        v = chr(u % 588 // 28 + 0x1161)
        t = chr(u % 28 + 0x11A7) if u % 28 else ""
        output += l + v + t
    return output

romaja, korean = Initialize(romaja), Initialize([decompose(word) for word in korean])
charset_max = max(len(romaja.charset), len(korean.charset))

def create_tensors(input: Initialize):
    for word in input.words:
        tensor = torch.zeros(max(romaja.max, korean.max), dtype=torch.long)
        for i, char in enumerate(word):
            tensor[i] = input.charset.index(char)
        input.tensors.append(tensor)

create_tensors(romaja)
create_tensors(korean)

class LSTM(nn.Module):
    def __init__(self, device):
        super(LSTM, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(charset_max, 256)
        self.lstm = nn.LSTM(256, 512, num_layers=2, batch_first=True, dropout=0.3)
        self.linear = nn.Linear(512, charset_max)
    
    def forward(self, input):
        hidden = (
            torch.zeros(2, input.size(0), 512).to(self.device),
            torch.zeros(2, input.size(0), 512).to(self.device)
        )
        output, hidden = self.lstm(self.embedding(input), hidden)
        return self.linear(output[:, -1, :])
    
model = LSTM(device).to(device)

dataset = TensorDataset(torch.stack(romaja.tensors), torch.stack(korean.tensors))
t, v = torch.utils.data.random_split(dataset, [
    int(len(dataset) * 0.8),
    len(dataset) - int(len(dataset) * 0.8)
])

training = DataLoader(t, batch_size=128, shuffle=True)
validation = DataLoader(v, batch_size=128, shuffle=False)
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=5)

def getElapsed():
    elapsed = timedelta(seconds=default_timer() - start)
    return str(elapsed).split(".")[0]

for epoch in range(50):
    model.train()
    training_loss = 0
    for r, k in training:
        r, k = r.to(device), k.to(device)
        optimizer.zero_grad()
        loss = criterion(model(r), k[:, 0])
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
            loss = criterion(model(r), k[:, 0])
            validation_loss += loss.item()
    
    avg_validation_loss = validation_loss / len(validation)
    scheduler.step(avg_validation_loss)
    
    print(f"[{getElapsed()}] Epoch {epoch + 1:02} of 50, Training Loss: {avg_training_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}")

torch.jit.script(model).save("out/model.pt")

print(f"\nFinished in {getElapsed()}")