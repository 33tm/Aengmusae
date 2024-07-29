import Link from "next/link"
import { Logo } from "@/components/logo"
import { Aengmusae } from "@/components/aengmusae"
import { FaGithub, FaInstagram, FaXTwitter } from "react-icons/fa6"

export default () => {
    return (
        <>
            <div className="pl-4 pt-4">
                <h1 className="text-3xl font-bold">Aengmusae (앵무새)</h1>
                <p>Convert romanized Korean text into Korean using AI</p>
            </div>
            <Aengmusae />
            <Link href="https://tttm.us">
                <Logo className="absolute bottom-12 left-12 h-24 fill-accent" />
            </Link>
            <div className="absolute flex flex-col justify-between h-24 bottom-12 left-44 space-y-2">
                <Link className="flex group" href="https://github.com/33tm">
                    <FaGithub size={24} />
                    <p className="pl-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">@33tm</p>
                </Link>
                <Link className="flex group" href="https://instagram.com/33tmmm">
                    <FaInstagram size={24} />
                    <p className="pl-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">@33tmmm</p>
                </Link>
                <Link className="flex group" href="https://x.com/33tmmm">
                    <FaXTwitter size={24} />
                    <p className="pl-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">@33tmmm</p>
                </Link>
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