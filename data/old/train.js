import { JSDOM } from "jsdom"
import { writeFileSync } from "fs"

let x = 0
let o = 0
const start = Date.now()

const getArtists = async () => {
    try {
        const { window } = await JSDOM.fromURL("https://colorcodedlyrics.com/index")
        return [...window.document
            .querySelectorAll(".entry-content a")]
            .filter(({ textContent }) => textContent.length > 1)
            .map(link => link.getAttribute("href"))
    } catch {
        return await getArtists()
    }
}

const getSongs = async artist => {
    try {
        const { window } = await JSDOM.fromURL(artist)
        return [...window.document
            .querySelectorAll("a[href^='https://colorcodedlyrics.com/20']:not(.post-thumbnail)")]
            .filter(({ children }) => !children.length)
    } catch {
        return
    }
}

const getLyrics = async song => {
    try {
        const { window } = await JSDOM.fromURL(song.getAttribute("href"))

        const tables = [...window.document
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
                const [romaja, korean] = [...window.document
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
            console.log("X", song.textContent)
            x++
            return
        }

        console.log("O", song.textContent)
        o++

        return romaja
            .map((word, i) => word !== korean[i] && { romaja: word, korean: korean[i] })
            .filter(Boolean)
            .map(({ romaja, korean }) => {
                let prefix = 0
                for (let i = 0; i < Math.min(romaja.length, korean.length); i++) {
                    if (romaja[i] === korean[i])
                        prefix++
                    else break
                }

                let suffix = 0
                for (let i = 0; i < Math.min(romaja.length, korean.length); i++) {
                    if (romaja[romaja.length - 1 - i] === korean[korean.length - 1 - i])
                        suffix++
                    else break
                }

                romaja = romaja.slice(prefix, romaja.length - suffix)
                korean = korean.slice(prefix, korean.length - suffix)

                if (romaja.match(/^[a-z]+$/) && korean.match(/^[가-힣]+$/))
                    return { romaja, korean }
            })
    } catch {
        return
    }

}

let words = []
const artists = await getArtists()

for (const artist in artists) {
    console.clear()
    console.log(`artist ${parseInt(artist) + 1} of ${artists.length}`)
    console.log(`${o}/${o + x} songs (${(o / (o + x) * 100).toFixed(2)}%)`)
    console.log(`${words.flat().filter(Boolean).length} words`)

    words.push(...await Promise.all(
        await getSongs(artists[artist])
            .then(songs => songs.map(getLyrics))
    ))
}

words = words.flat().filter(Boolean)

writeFileSync("train.json", JSON.stringify(words))

console.clear()
console.log(`${words.length} words`)
console.log(`${o}/${o + x} songs (${(o / (o + x) * 100).toFixed(2)}%)`)
console.log(`${new Date(Date.now() - start).toISOString().slice(11, -5)} elapsed`)