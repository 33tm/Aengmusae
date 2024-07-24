from re import sub
from os import mkdir
from csv import writer
from json import loads
from requests import get
from os.path import exists

if exists("temp/dictionary.jsonl"):
    with open("temp/dictionary.jsonl", encoding="utf-8") as file:
        dictionary = file.read().splitlines()
else:
    res = get("https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.jsonl").text
    dictionary = res.splitlines()
    if not exists("temp"):
        mkdir("temp")
    with open("temp/dictionary.jsonl", "w", encoding="utf-8") as file:
        file.write(res)

words = set()
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
        words.add((romaja, korean))

with open("test.csv", "w", encoding="utf-8", newline="") as file:
    writer(file).writerows(words)
    
print(f"downloaded {len(words)} pairs")