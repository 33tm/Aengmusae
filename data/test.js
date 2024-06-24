import { writeFileSync } from "fs"

const start = Date.now()

const dictionary = await fetch("https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.json")
    .then(res => res.text())
    .then(dictionary => dictionary
        .split("\n")
        .filter(Boolean)
        .map(JSON.parse)
        .filter(({ forms }) => forms)
        .flatMap(({ forms }) => forms
            .filter(({ form, roman }) => form && roman)
            .flatMap(({ form, roman }) => {
                if (roman.includes("/")) return

                let romaja = roman.replace(/[^a-z가-힣\s/]+/g, "").split(" ")
                let korean = form.replace(/[^a-z가-힣\s]+/g, "").split(" ")

                if (romaja.length !== korean.length) return

                return romaja.map((romaja, i) => ({ romaja, korean: korean[i] }))
            }))
        .filter(Boolean))

writeFileSync("test.json", JSON.stringify(dictionary))

console.log(`Fetched ${dictionary.length} words in ${Math.round((Date.now() - start) / 1000)} seconds.`)