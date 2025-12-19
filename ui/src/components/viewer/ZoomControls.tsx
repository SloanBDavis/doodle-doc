import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { ZoomIn, ZoomOut, Maximize, ArrowLeftRight } from "lucide-react";

interface ZoomControlsProps {
  zoom: number;
  fitMode: "width" | "page" | "none";
  onZoomChange: (zoom: number) => void;
  onFitModeChange: (mode: "width" | "page" | "none") => void;
}

export function ZoomControls({
  zoom,
  fitMode,
  onZoomChange,
  onFitModeChange,
}: ZoomControlsProps) {
  return (
    <div className="flex items-center gap-2">
      <Button
        variant="outline"
        size="icon"
        onClick={() => {
          onFitModeChange("none");
          onZoomChange(Math.max(0.25, zoom - 0.25));
        }}
      >
        <ZoomOut className="size-4" />
      </Button>
      <span className="text-sm w-16 text-center">{Math.round(zoom * 100)}%</span>
      <Button
        variant="outline"
        size="icon"
        onClick={() => {
          onFitModeChange("none");
          onZoomChange(Math.min(3, zoom + 0.25));
        }}
      >
        <ZoomIn className="size-4" />
      </Button>
      <ToggleGroup
        type="single"
        value={fitMode}
        onValueChange={(v) => {
          if (v) onFitModeChange(v as "width" | "page" | "none");
        }}
      >
        <ToggleGroupItem value="width" aria-label="Fit to width">
          <ArrowLeftRight className="size-4" />
        </ToggleGroupItem>
        <ToggleGroupItem value="page" aria-label="Fit to page">
          <Maximize className="size-4" />
        </ToggleGroupItem>
      </ToggleGroup>
    </div>
  );
}
