"use client"

import { FormEvent, useState } from "react"
import { CgSpinner } from "react-icons/cg"

export const Aengmusae = () => {
    const [input, setInput] = useState("")
    const [output, setOutput] = useState("")
    const [error, setError] = useState(false)
    const [loading, setLoading] = useState(false)

    const submit = (event: FormEvent) => {
        event.preventDefault()

        if (!/^[a-z\s]*$/.test(input.toLowerCase())) {
            setOutput("Invalid input!")
            return
        }

        setLoading(true)

        fetch(process.env.NEXT_PUBLIC_API_URL!, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: input || "annyeong" })
        })
            .then(async res => setOutput(await res.text()))
            .catch(() => setError(true))
            .then(() => setLoading(false))
    }

    return (
        <div className="flex w-full mt-4">
            {error ? (
                <p className="m-auto text-xl font-semibold">
                    Service unavailable, server offline.
                </p>
            ) : (
                <>
                    <form onSubmit={submit} className={`flex flex-col md:flex-row m-auto ${loading && "opacity-80 cursor-not-allowed"}`}>
                        <input
                            type="text"
                            className="bg-background p-2 border-2 border-accent text-center md:text-left rounded-t-lg md:rounded-none md:rounded-l-lg focus:outline-none"
                            placeholder="annyeong"
                            onChange={({ target }) => setInput(target.value)}
                            disabled={loading}
                        />
                        <button
                            type="submit"
                            className="bg-accent p-2 md:pr-3 border-2 border-accent rounded-b-lg md:rounded-none md:rounded-r-lg text-background"
                            disabled={loading}
                        >
                            Submit
                        </button>
                    </form>
                    <p className="absolute w-[calc(100vw-4rem)] mt-28 text-4xl text-center font-semibold">
                        {loading ? (
                            <CgSpinner className="animate-spin m-auto" />
                        ) : (
                            <>{output}</>
                        )}
                    </p>
                </>
            )}
        </div >
    )
}