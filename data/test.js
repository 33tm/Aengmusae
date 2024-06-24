import { writeFileSync } from "fs"

const start = Date.now()

const dictionary = await fetch("https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.json")
    .then(res => res.text())
    .then(dictionary => dictionary
        .split("\n")
        .filter(Boolean)
        .map(JSON.parse)
        .filter(({ forms }) => forms)
        .map(({ forms }) => {
            const obj = {}

            forms.forEach(({ form, tags }) => {
                if (!tags) return
                obj[tags[0]] = form
            })

            const romaja = obj.romanization?.toLowerCase()
            const korean = obj.hangeul

            if (!romaja || !korean || `${romaja}${korean}`.match(/[^a-z가-힣]+/)) return

            return { romaja, korean }
        })
        .filter(Boolean))

writeFileSync("test.json", JSON.stringify(dictionary))

console.log(`Fetched ${dictionary.length} words in ${Math.round((Date.now() - start) / 1000)} seconds.`)