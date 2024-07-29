import "@/app/style.css"
import { Inter } from "next/font/google"
import type { Metadata } from "next"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
    title: "Aengmusae - Convert Romanized Korean to Korean",
    description: "Convert romanized Korean text into Korean Hangul using AI",
    keywords: [
        "korean romanization to korean",
        "korean romanization to hangul",
        "romanized korean to korean",
        "romanized korean to hangul",
        "romanization to hangul",
        "romanization to korean",
        "romaja to hangul",
        "romaja to korean",
        "reverse korean romanization",
        "romanized korean converter",
        "convert romanized korean",
        "korean nlp",
        "korean ai",
        "romaja"
    ]
}

export default ({ children }: Readonly<{ children: React.ReactNode }>) => {
    return (
        <html lang="en">
            <body className={`${inter.className} m-4 h-[calc(100vh-2rem)] outline outline-4 bg-background text-accent`}>
                <script
                    type="application/ld+json"
                    dangerouslySetInnerHTML={{
                        __html: JSON.stringify({
                            "@context": "https://schema.org",
                            "@type": "WebSite",
                            name: "Aengmusae",
                            url: "https://aengmusae.tttm.us/"
                        })
                    }}
                />
                {children}
            </body>
        </html>
    )
}