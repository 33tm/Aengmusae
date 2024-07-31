import Link from "next/link"
import { Logo } from "@/components/logo"
import { Aengmusae } from "@/components/aengmusae"
import { FaGithub, FaInstagram, FaXTwitter } from "react-icons/fa6"

export default () => {
    return (
        <>
            <div className="absolute flex justify-between z-10 w-[calc(100vw-4rem)] md:w-auto top-12 left-12 md:top-auto md:bottom-12">
                <Link href="https://tttm.us">
                    <Logo className="h-12 md:h-24 fill-accent" />
                </Link>
                <div className="flex md:flex-col my-auto md:m-0 pr-8 md:pr-0 md:w-auto md:ml-8 md:justify-between space-x-4 md:space-x-0 md:space-y-2">
                    <Link href="https://github.com/33tm">
                        <FaGithub
                            size={24}
                            className="hover:opacity-80 transition-opacity duration-200"
                        />
                    </Link>
                    <Link href="https://instagram.com/33tmmm">
                        <FaInstagram
                            size={24}
                            className="hover:opacity-80 transition-opacity duration-200"
                        />
                    </Link>
                    <Link href="https://x.com/33tmmm">
                        <FaXTwitter
                            size={24}
                            className="hover:opacity-80 transition-opacity duration-200"
                        />
                    </Link>
                </div>
            </div>
            <div className="absolute flex flex-col h-[calc(100vh-2rem)] w-[calc(100vw-2rem)] align-middle text-center">
                <div className="m-auto text-center mx-4">
                    <h1 className="text-3xl font-bold">Aengmusae (앵무새)</h1>
                    <p>Convert romanized Korean text into Korean Hangul using AI</p>
                    <Aengmusae />
                </div>
            </div>
            <div className="absolute bottom-12 right-12 text-right">
                <p>
                    Trained on lyrics from
                    {" "}
                    <Link
                        href="https://colorcodedlyrics.com"
                        className="font-semibold hover:underline"
                    >
                        colorcodedlyrics.com
                    </Link>
                </p>
                <p>
                    More information available at
                    {" "}
                    <Link
                        href="https://github.com/33tm/Aengmusae"
                        className="font-semibold hover:underline"
                    >
                        github.com/33tm/aengmusae
                    </Link>
                </p>
            </div>
        </>
    )
}