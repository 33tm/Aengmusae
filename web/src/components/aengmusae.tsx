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
        }).catch(() => setError(true))
    }

    return (
        <div className="flex w-full mt-4">
            {error ? (
                <p className="m-auto text-xl font-semibold">
                    Service unavailable, server offline.
                </p>
            ) : (
                <>
                    <form onSubmit={submit} className="flex flex-col md:flex-row m-auto">
                        <input
                            type="text"
                            className="bg-background p-2 border-2 border-accent rounded-t-lg md:rounded-none md:rounded-l-lg focus:outline-none"
                            placeholder="annyeong"
                            onChange={({ target }) => setInput(target.value)}
                        />
                        <button
                            type="submit"
                            className="bg-accent p-2 md:pr-3 border-2 border-accent rounded-b-lg md:rounded-none md:rounded-r-lg text-background"
                        >
                            Submit
                        </button>
                    </form>
                    <p className="absolute w-[calc(100vw-4rem)] mt-28 text-4xl text-center font-semibold">
                        {output}
                    </p>
                </>
            )}
        </div >
    )
}