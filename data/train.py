from os import mkdir
from csv import writer
from requests import get
from bs4 import BeautifulSoup
from datetime import timedelta
from timeit import default_timer
from os.path import exists, commonprefix

start = default_timer()

def getElapsed():
    elapsed = timedelta(seconds=default_timer() - start)
    return str(elapsed).split('.')[0]

def getSongs(url):
    soup = BeautifulSoup(get(url).content, "html.parser")
    songs = {a["href"] for a in soup.find_all("a", rel="bookmark")}
    next = soup.find("div", class_="nav-previous")
    if next:
        page = next.find("a")["href"]
        print(f"page {page.split('/')[-2]}, {getElapsed()} elapsed")
        songs.update(getSongs(page))
    return list(songs)

def getLyrics(url):
    soup = BeautifulSoup(get(url).content, "html.parser")
    table = soup.find("table", border=0)
    head = [item.getText() for item in table.findAll("th")]
    body = [item.getText() for item in table.findAll("td")]
    try:
        romaja = body[head.index("Romanization")].split("\n")
        korean = body[head.index("Korean")].split("\n")
        if len(romaja) != len(korean):
            return
        lyrics = []
        for r, k in zip(romaja, korean):
            if r == k:
                continue
            prefix_len = len(commonprefix([r, k]))
            r, k = r[prefix_len:], k[prefix_len:]
            suffix_len = len(commonprefix([r[::-1], k[::-1]]))
            if suffix_len > 0:
                r, k = r[:-suffix_len], k[:-suffix_len]
            lyrics.append((r, k))
        return lyrics
    except:
        return

if exists("data/temp/songs.txt"):
    with open("data/temp/songs.txt", "r") as file:
        songs = file.read().split("\n")
else:
    songs = getSongs("https://colorcodedlyrics.com/category/krn/page/1450")
    if not exists("data/temp"):
        mkdir("data/temp")
    with open("data/temp/songs.txt", "w") as file:
        file.write(("\n").join(songs))

print(f"loaded {len(songs)} songs")

rows = getLyrics(songs[0])
with open("data/train.csv", "w") as file:
    csv = writer(file)
    csv.writerow(["romaja", "korean"])
    csv.writerows(rows)