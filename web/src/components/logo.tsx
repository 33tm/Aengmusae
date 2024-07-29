export const Logo = ({ className }: { className?: string }) => {
    return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" className={className}>
            <path shapeRendering="crispEdges" d="M0,0h16v4H0V0Z" />
            <path shapeRendering="crispEdges" d="M16,0h16v4h-16V0Z" />
            <path shapeRendering="crispEdges" d="M22,24V4h4v20h-4Z" />
            <path shapeRendering="crispEdges" d="M6,16V4h4v12h-4Z" />
            <path shapeRendering="crispEdges" d="M0,16h16v4H0v-4Z" />
            <path shapeRendering="crispEdges" d="M6,32v-12h4v12h-4Z" />
            <path shapeRendering="crispEdges" d="M16,32v-16h4v16h-4Z" />
            <path shapeRendering="crispEdges" d="M28,32v-16h4v16h-4Z" />
        </svg>
    )
}