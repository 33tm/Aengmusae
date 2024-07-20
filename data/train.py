from re import match
from csv import writer
from os import makedirs
from requests import get
from bs4 import BeautifulSoup
from datetime import timedelta
from timeit import default_timer
from os.path import commonprefix, exists

start = default_timer()

def trimDuplicate(a, b):
    prefix_len = len(commonprefix([a, b]))
    suffix_len = len(commonprefix([a[::-1], b[::-1]]))
    a = a[prefix_len:-suffix_len] if suffix_len > 0 else a[prefix_len:]
    b = b[prefix_len:-suffix_len] if suffix_len > 0 else b[prefix_len:]
    return a, b

def getElapsed():
    elapsed = timedelta(seconds=default_timer() - start)
    return str(elapsed).split('.')[0]

def getSongs(url):
    soup = BeautifulSoup(get(url).content, "html.parser")
    songs = {a["href"] for a in soup.find_all("a", rel="bookmark")}
    next = soup.find("div", class_="nav-previous")
    print(f"page {url.split('/')[-2]}, {getElapsed()} elapsed")
    if next:
        page = next.find("a")["href"]
        songs.update(getSongs(page))
    return list(songs)

def getLyrics(url):
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

    romaja = body[head.index("Romanization")]
    korean = body[head.index("Korean") if "Korean" in head else head.index("Hangul")]

    romaja = [word for word in romaja.lower().split("\n") if word]
    korean = [word for word in korean.lower().split("\n") if word]

    if len(romaja) != len(korean):
        return []
    
    lyrics = []
    for r, k in zip(romaja, korean):
        r, k = trimDuplicate(r.strip(), k.strip())
        if not r or not k:
            continue
        r, k = r.split(" "), k.split(" ")
        if len(r) != len(k):
            continue
        r_line, k_line = [], []
        for i in range(len(r)):
            r[i], k[i] = trimDuplicate(r[i], k[i])
            r[i] = r[i].replace("-", "")
            if match(r"^[a-z\s]+$", r[i]) and match(r"^[가-힣\s]+$", k[i]):
                r_line.append(r[i])
                k_line.append(k[i])
        lyrics.append((" ".join(r_line), " ".join(k_line)))
    return lyrics

if exists("data/temp/songs.txt"):
    with open("data/temp/songs.txt", "r") as file:
        songs = file.read().split("\n")
else:
    songs = getSongs("https://colorcodedlyrics.com/category/krn/page/1/")
    makedirs("data/temp", exist_ok=True)
    with open("data/temp/songs.txt", "w") as file:
        file.write(("\n").join(songs))

print(f"loaded {len(songs)} songs in {getElapsed()}")

rows = []

print(f"scraped {len(rows)} pairs in {getElapsed()}")
with open("data/train.csv", "w") as file:
    csv = writer(file)
    csv.writerow(["romaja", "korean"])
    csv.writerows(rows)