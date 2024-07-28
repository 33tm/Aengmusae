from csv import reader
from waitress import serve
from torch import torch, nn
from jamo import h2j, j2hcj, j2h
from flask import Flask, request

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("out/data.csv", encoding="utf-8", newline="") as file:
    data = [tuple(row) for row in reader(file)]
    romaja, korean = zip(*data)

class Initialize:
    def __init__(self, words):
        self.charset = [" "] + sorted(list(set("".join(words))))
        self.max = max([len(word) for word in words])

def decompose(word):
    return ".".join([j2hcj(h2j(syllable)) for syllable in word])

romaja, korean = map(Initialize, [romaja, [decompose(word) for word in korean]])
charset_max = max(len(romaja.charset), len(korean.charset))

class LSTM(nn.Module):
    def __init__(self, device):
        super(LSTM, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(0.3)
        self.embedding = nn.Embedding(charset_max, 256, padding_idx=0)
        self.lstm = nn.LSTM(256, 512, num_layers=3, batch_first=True, bidirectional=True, dropout=0.3)
        self.linear = nn.Linear(1024, charset_max)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        hidden = (  
            torch.zeros(6, input.size(0), 512).to(self.device),
            torch.zeros(6, input.size(0), 512).to(self.device)
        )
        output, _ = self.lstm(embedded, hidden)
        return self.linear(output)

model = LSTM(device).to(device)
model.load_state_dict(torch.load("out/model.pt", map_location=device))
model.eval()

app = Flask(__name__)

@app.route("/", methods=["POST"])
def evaluate():
    input, output = request.json["query"].lower(), []
    for word in input.split():
        tensor = torch.tensor([romaja.charset.index(char) for char in word]).to(device)
        with torch.no_grad():
            indexes = torch.argmax(model(tensor.unsqueeze(0)), dim=-1).squeeze(0).tolist()
        jamos = [c for c in "".join([korean.charset[i] for i in indexes]).split(".")]
        temp = ""
        for jamo in jamos:
            try:
                temp += j2h(*jamo)
            except:
                temp += jamo
        output.append(temp)
    return " ".join(output)

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)