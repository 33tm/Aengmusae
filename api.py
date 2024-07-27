from flask import Flask, request

import torch
from csv import reader
from json import loads
from jamo import h2j, j2hcj, j2h

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

model = torch.jit.load("out/model.pt")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

@app.route("/", methods=["POST"])
def evaluate():
    input, output = request.json["query"], []
    for word in input.split():
        tensor = torch.tensor([romaja.charset.index(char) for char in word]).to(device)
        with torch.no_grad():
            indexes = torch.argmax(model(tensor.unsqueeze(0)), dim=-1).squeeze(0).tolist()
        jamos = [c for c in "".join([korean.charset[i] for i in indexes]).split(".") if c]
        temp = ""
        for jamo in jamos:
            try:
                temp += j2h(*jamo)
            except:
                temp += jamo
        output.append(temp.replace(" ", ""))
    return " ".join(output)

if __name__ == "__main__":
    app.run()