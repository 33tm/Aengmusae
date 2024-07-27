import torch
from csv import reader
from jamo import h2j, j2hcj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("out/model.pt")
model.eval()

with open("out/data.csv", encoding="utf-8", newline="") as file:
    data = [tuple(row) for row in reader(file)][:1000]
    romaja, korean = zip(*data)

class Initialize:
    def __init__(self, words):
        self.charset = [" "] + sorted(list(set("".join(words))))
        self.max = max([len(word) for word in words])

def decompose(word):
    return ".".join([j2hcj(h2j(syllable)) for syllable in word])

romaja, korean = map(Initialize, [romaja, [decompose(word) for word in korean]])

tensor = torch.zeros(max(romaja.max, korean.max), dtype=torch.long).to(device)
for i, char in enumerate("na"):
    tensor[i] = romaja.charset.index(char)

with torch.no_grad():
    indexes = torch.argmax(model(tensor.unsqueeze(0)), dim=-1).squeeze(0).tolist()
    print([i for i in indexes])
    print("".join([korean.charset[i] for i in indexes]))