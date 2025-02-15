from os import mkdir
from csv import writer
from json import loads
from requests import get
from re import match, sub
from bs4 import BeautifulSoup
from datetime import timedelta
from timeit import default_timer
from os.path import commonprefix, exists
from concurrent.futures import ThreadPoolExecutor, as_completed

start = default_timer()

def getElapsed():
    elapsed = timedelta(seconds=default_timer() - start)
    return str(elapsed).split(".")[0]

def getSongs(page):
    try:
        url = f"https://colorcodedlyrics.com/category/krn/page/{page}"
        soup = BeautifulSoup(get(url).content, "html.parser")
        songs = {a["href"] for a in soup.find_all("a", rel="bookmark")}
        print(f"{getElapsed()} - page {page}", end="\r")
        return list(songs)
    except Exception as e:
        print(f"{e.__class__.__name__} - {url}")
        return []

def getLyrics(url):
    try:
        soup = BeautifulSoup(get(url).content, "html.parser")
        head = [
            item.getText() for item in
            soup.select("table[border='0'] th")
            or
            soup.select(".wp-block-column > p > strong > span")
            or
            soup.select("table")[1].select("th")
        ]
        body = [
            item.getText() for item in
            soup.select("table[border='0'] td")
            or
            soup.select(".wp-block-column > .wp-block-group > div")
            or
            soup.select("table")[1].select("td")
        ]

        if len(head) != len(body) or "Romanization" not in head or ("Korean" not in head and "Hangul" not in head):
            return []

        romaja = body[head.index("Romanization")]
        korean = body[head.index("Korean") if "Korean" in head else head.index("Hangul")]

        romaja = [word for word in romaja.lower().splitlines() if word]
        korean = [word for word in korean.lower().splitlines() if word]

        if len(romaja) != len(korean):
            return []
        
        words = set()
        for r, k in zip(romaja, korean):
            r, k = r.strip(), k.strip()
            pl = len(commonprefix([r, k]))
            sl = len(commonprefix([r[::-1], k[::-1]]))
            r = r[pl:-sl] if sl > 0 else r[pl:]
            k = k[pl:-sl] if sl > 0 else k[pl:]
            if not r or not k:
                continue
            r, k = r.split(), k.split()
            if len(r) != len(k):
                continue
            for rw, kw in zip(r, k):
                if match(r"^[a-z\s]+$", rw) and match(r"^[가-힣\s]+$", kw):
                    words.add((rw, kw))
        print(f"{getElapsed()} - {url.split('/')[-2].replace('-', ' ')}", end="\r")
        return words
    except Exception as e:
        print(f"{e.__class__.__name__} - {url}")
        return []
    
if not exists("out"):
    mkdir("out")

if exists("out/songs.txt"):
    with open("out/songs.txt", "r") as file:
        songs = file.read().splitlines()
else:
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(getSongs, i) for i in range(1465)]
    songs = {song for future in as_completed(futures) for song in future.result()}
    if len(songs):
        with open("out/songs.txt", "w") as file:
            file.write("\n".join(songs))

with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(getLyrics, song) for song in songs]
words = {word for future in as_completed(futures) for word in future.result()}

if exists("out/dictionary.jsonl"):
    with open("out/dictionary.jsonl", encoding="utf-8") as file:
        dictionary = file.read().splitlines()
else:
    res = get("https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.jsonl").text
    dictionary = res.splitlines()
    with open("out/dictionary.jsonl", "w", encoding="utf-8") as file:
        file.write(res)

for word in dictionary:
    if not word:
        continue
    word = loads(word)
    if "forms" not in word:
        continue
    for form in word["forms"]:
        if "roman" not in form or "form" not in form:
            continue
        romaja = sub(r"[^a-z\s]+", "", form["roman"])
        korean = sub(r"[^가-힣\s]", "", form["form"])
        for r, k in zip(romaja.split(), korean.split()):
            words.add((r, k))

with open("out/data.csv", "w", encoding="utf-8", newline="") as file:
    writer(file).writerows(words)

print(f"\nscraped {len(words)} pairs ({len(songs)} songs) in {getElapsed()}")