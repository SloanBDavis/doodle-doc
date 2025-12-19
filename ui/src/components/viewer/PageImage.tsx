import type { CSSProperties } from "react";

interface PageImageProps {
  src: string;
  zoom: number;
  fitMode: "width" | "page" | "none";
}

export function PageImage({ src, zoom, fitMode }: PageImageProps) {
  const style: CSSProperties =
    fitMode === "none"
      ? { transform: `scale(${zoom})`, transformOrigin: "top center" }
      : {};

  const className =
    fitMode === "width"
      ? "w-full h-auto"
      : fitMode === "page"
        ? "max-w-full max-h-full object-contain"
        : "";

  return (
    <div className="flex-1 overflow-auto flex justify-center p-4">
      <img
        src={src}
        alt="Page"
        className={className}
        style={style}
        loading="eager"
      />
    </div>
  );
}
