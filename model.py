from csv import reader
from os.path import exists
from datetime import timedelta
from timeit import default_timer
from torch import torch, nn, optim

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
            raise Exception(f'Non-Korean syllable "{syllable}" in word "{word}"')
        l = chr(u // 588 + 0x1100)
        v = chr(u % 588 // 28 + 0x1161)
        t = chr(u % 28 + 0x11A7) if u % 28 else ""
        output += l + v + t
    return output

romaja, korean = Initialize(romaja), Initialize([decompose(word) for word in korean])

charset_max = max(len(romaja.charset), len(korean.charset))

class RNN(nn.Module):
    def __init__(self, device):
        super(RNN, self).__init__()
        self.device = device
        self.rnn = nn.RNN(charset_max, 128, batch_first=True)
        self.linear = nn.Linear(128, charset_max)
    
    def forward(self, input):
        hidden = torch.zeros(1, input.size(0), 128).to(self.device)
        output, hidden = self.rnn(input, hidden)
        return self.linear(output[:, -1, :])
    
model = RNN(device).to(device)
    
def create_tensors(input: Initialize):
    for word in input.words:
        tensor = torch.zeros(
            max(romaja.max, korean.max),
            1,
            charset_max,
        )
        for i, char in enumerate(word):
            tensor[i][0][input.charset.index(char)] = 1
        input.tensors.append(tensor)

create_tensors(romaja)
create_tensors(korean)
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    total_loss = 0
    for r, k in zip(romaja.tensors, korean.tensors):
        r, k = r.to(device), k.to(device)
        optimizer.zero_grad()
        loss = criterion(model(r), k.squeeze())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{100}], Loss: {(total_loss / len(romaja.tensors)):.4f}')

torch.jit.script(model).save("out/model.pt")

print(f"\nFinished in {str(timedelta(seconds=default_timer() - start)).split(".")[0]}")