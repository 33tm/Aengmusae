"use client"

import { FormEvent, useState } from "react"

export const Aengmusae = () => {
    const [input, setInput] = useState("")
    const [output, setOutput] = useState("")
    const [error, setError] = useState(false)

    const submit = (event: FormEvent) => {
        event.preventDefault()
        if (!/^[a-z\s]*$/.test(input.toLowerCase())) {
            setOutput("Invalid input!")
            return
        }
        fetch(process.env.NEXT_PUBLIC_API_URL!, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: input || "annyeong" })
        }).then(async res => {
            if (!res.ok) setError(true)
            else setOutput(await res.text())
        })
    }

    if (error) {
        return (
            <div className="flex">
                <p className="m-auto text-2xl pt-8 font-semibold">
                    Service unavailable, API is offline.
                </p>
            </div>
        )
    }

    return (
        <>
            <form onSubmit={submit}>
                <input
                    type="text"
                    className="bg-background border-2 border-accent rounded-l-lg p-2 ml-4 mt-4 focus:outline-none"
                    placeholder="annyeong"
                    onChange={({ target }) => setInput(target.value)}
                />
                <button
                    type="submit"
                    className="p-2 pr-3 bg-accent border-2 border-accent rounded-r-lg text-background"
                >
                    Submit
                </button>
            </form>
            <p className="p-4 text-4xl font-semibold">{output}</p>
        </>
    )
}