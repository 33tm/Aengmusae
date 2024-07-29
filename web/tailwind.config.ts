import type { Config } from "tailwindcss"

export default { 
    content: ["./src/**/*.tsx"],
    theme: {
        extend: {
            colors: {
                background: "var(--background)",
                accent: "var(--accent)"
            }
        }
    }
} satisfies Config