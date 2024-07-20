from re import sub
from os import mkdir
from csv import writer
from json import loads
from requests import get
from os.path import exists

if exists("temp/dictionary.jsonl"):
    with open("temp/dictionary.jsonl") as file:
        dictionary = file.read().split("\n")
else:
    res = get("https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.jsonl").text
    dictionary = res.split("\n")
    if not exists("temp"):
        mkdir("temp")
    with open("temp/dictionary.jsonl", "w") as file:
        file.write(res)

words = []
for word in dictionary:
    if not word:
        continue
    word = loads(word)
    if not "forms" in word:
        continue
    for form in word["forms"]:
        if not "roman" in form or not "form" in form:
            continue
        romaja = sub(r"[^a-z\s]+", "", form["roman"])
        korean = sub(r"[^가-힣\s]", "", form["form"])
        words.append((romaja, korean))

print(f"generated {len(words)} pairs")
with open("test.csv", "w") as file:
    csv = writer(file)
    csv.writerow(["romaja", "korean"])
    csv.writerows(words)