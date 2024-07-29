import "@/app/style.css"
import { Inter } from "next/font/google"
import type { Metadata } from "next"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
    title: "Aengmusae (앵무새)",
    description: "Convert romanized Korean text into Korean using AI"
}

export default ({ children }: Readonly<{ children: React.ReactNode }>) => {
    return (
        <html lang="en">
            <body className={`${inter.className} m-4 h-[calc(100vh-2rem)] outline outline-4 bg-background text-accent`}>
                {children}
            </body>
        </html>
    )
}