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
        
        lyrics = []
        for r, k in zip(romaja, korean):
            r, k = r.strip(), k.strip()
            prefix_len = len(commonprefix([r, k]))
            suffix_len = len(commonprefix([r[::-1], k[::-1]]))
            r = r[prefix_len:-suffix_len] if suffix_len > 0 else r[prefix_len:]
            k = k[prefix_len:-suffix_len] if suffix_len > 0 else k[prefix_len:]
            if not r or not k:
                continue
            r, k = r.split(), k.split()
            if len(r) != len(k):
                continue
            r = " ".join([word for word in r if match(r"^[a-z\s]+$", word)])
            k = " ".join([word for word in k if match(r"^[가-힣\s]+$", word)])
            if r and k:
                lyrics.append((r, k))
        print(f"{getElapsed()} - {url.split('/')[-2].replace('-', ' ')}", end="\r")
        return lyrics
    except Exception as e:
        print(f"{e.__class__.__name__} - {url}")
        return []

if exists("temp/songs.txt"):
    with open("temp/songs.txt", "r") as file:
        songs = file.read().splitlines()
else:
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(getSongs, i) for i in range(1465)]
    songs = {song for future in as_completed(futures) for song in future.result()}
    if not exists("temp"):
        mkdir("temp")
    if len(songs):
        with open("temp/songs.txt", "w") as file:
            file.write("\n".join(songs))

with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(getLyrics, song) for song in songs]
lyrics = {lyric for future in as_completed(futures) for lyric in future.result()}

with open("train.csv", "w", encoding="utf-8", newline="") as file:
    csv = writer(file)
    csv.writerow(["romaja", "korean"])
    csv.writerows(lyrics)

print(f"\nscraped {len(lyrics)} pairs in {getElapsed()}")