import { JSDOM } from "jsdom"
import { writeFileSync } from "fs"

let x = 0
let o = 0
const start = Date.now()

const getLyrics = async link => {
    const { window } = await JSDOM.fromURL(link.getAttribute("href"))
    const { document } = window

    const tables = [...document
        .querySelectorAll("table[border='0'] tr")]
        .map(({ children }) => [...children].map(({ textContent }) => textContent.toLowerCase()))

    const { romaja, korean } = (() => {
        if (tables.length) {
            const [head, body] = tables
            return Object.fromEntries(head.map((key, i) => [
                key === "romanization" ? "romaja" : key.trim(),
                body[i].split(/[\s()\[\]]+/)
            ]))
        } else {
            const [romaja, korean] = [...document
                .querySelectorAll(".wp-block-group__inner-container .wp-block-spacer")]
                .map(({ parentNode }) => parentNode
                    .querySelector(".wp-block-group__inner-container")
                    .textContent
                    .trim()
                    .split(/[\s()\[\]]+/))
            return { romaja, korean }
        }
    })()

    if (!romaja || !korean || romaja.length !== korean.length) {
        console.log("X", link.textContent)
        x++
        return null
    }

    console.log("O", link.textContent)
    o++

    return romaja
        .map((word, i) => {
            if (word === korean[i]) return null
            return { romaja: word, korean: korean[i] }
        })
        .filter(Boolean)
        .map(({ romaja, korean }) => {
            let prefix = 0
            for (let i = 0; i < Math.min(romaja.length, korean.length); i++) {
                if (romaja[i] === korean[i]) prefix++
                else break
            }
            let suffix = 0
            for (let i = 0; i < Math.min(romaja.length, korean.length); i++) {
                if (romaja[romaja.length - 1 - i] === korean[korean.length - 1 - i]) suffix++
                else break
            }
            romaja = romaja.slice(prefix, romaja.length - suffix)
            korean = korean.slice(prefix, korean.length - suffix)
            if (korean.match(/[가-힣]+/)) return { romaja, korean }
        })
}

const getSongs = async link => JSDOM.fromURL(link.getAttribute("href"))
    .then(({ window }) => [...window.document
        .querySelectorAll("a[href^='https://colorcodedlyrics.com/20']:not(.post-thumbnail)")]
        .filter(({ children }) => !children.length)
        .map(getLyrics))

const artists = await JSDOM.fromURL("https://colorcodedlyrics.com/index")
    .then(({ window }) => [...window.document
        .querySelectorAll(".entry-content a")]
        .filter(({ textContent }) => textContent.length > 1))

let words = []

for (const artist in artists) {
    console.clear()
    console.log(`${new Date(Date.now() - start).toISOString().slice(11, -5)} elapsed`)
    console.log(`${artists[artist].textContent} (${Math.round(parseInt(artist) + 1)}/${artists.length})`)
    console.log(`${o + x} songs (${words.flat().filter(Boolean).length} words)`)
    const songs = await Promise.allSettled(await getSongs(artists[artist]))
        .then(words => words.flatMap(({ value }) => value))
    words.push(songs)
}

words = words.flat().filter(Boolean)

writeFileSync("train.json", JSON.stringify(words))

console.clear()

console.log("O", o)
console.log("X", x)
console.log((o / (o + x) * 100).toFixed(2) + "%")
console.log(`Scraped ${o + x} songs (${words.length} words) in ${Math.round((Date.now() - start) / 1000)} seconds.`)