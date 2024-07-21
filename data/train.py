from os import mkdir
from re import match
from csv import writer
from requests import get
from bs4 import BeautifulSoup
from datetime import timedelta
from timeit import default_timer
from os.path import commonprefix, exists
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def getSongs(page):
    try:
        url = f"https://colorcodedlyrics.com/category/krn/page/{page}"
        soup = BeautifulSoup(get(url).content, "html.parser")
        songs = {a["href"] for a in soup.find_all("a", rel="bookmark")}
        print(f"{getElapsed()} - page {page}", end="\r")
        return list(songs)
    except Exception as e:
        if e.__class__.__name__ == "ConnectionError":
            return getSongs(page)
        print(f"{e} - {url}")
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

        if not "Romanization" in head or (not "Korean" in head and not "Hangul" in head):
            return []

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
        print(f"{getElapsed()} - {url.split('/')[-2].replace('-', ' ')}", end="\r")
        return lyrics
    except Exception as e:
        if e.__class__.__name__ == "ConnectionError":
            return getLyrics(url)
        print(f"{e} - {url}")
        return []

if exists("temp/songs.txt"):
    with open("temp/songs.txt", "r") as file:
        songs = file.read().split("\n")
else:
    songs = []
    futures = []
    # 1465 pages in the "Korean" category of CCL
    with ThreadPoolExecutor(max_workers=100) as executor:
        for i in range(1465):
            futures.append(executor.submit(getSongs, i))
    for i, future in enumerate(as_completed(futures)):
        songs += future.result()
    if not exists("temp"):
        mkdir("temp")
    with open("temp/songs.txt", "w") as file:
        file.write("\n".join(songs))

lyrics = []
futures = []
with ThreadPoolExecutor(max_workers=100) as executor:
    for song in songs:
        futures.append(executor.submit(getLyrics, song))
for i, future in enumerate(as_completed(futures)):
    lyrics += future.result()

with open("train.csv", "w") as file:
    csv = writer(file)
    csv.writerow(["romaja", "korean"])
    csv.writerows(lyrics)

print(f"scraped {len(lyrics)} pairs in {getElapsed()}")