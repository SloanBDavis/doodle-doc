import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Pen, Eraser, Trash2 } from "lucide-react";

interface ToolBarProps {
  activeTool: "pen" | "eraser";
  strokeWidth: number;
  onToolChange: (tool: "pen" | "eraser") => void;
  onStrokeWidthChange: (width: number) => void;
  onClear: () => void;
}

export function ToolBar({
  activeTool,
  strokeWidth,
  onToolChange,
  onStrokeWidthChange,
  onClear,
}: ToolBarProps) {
  return (
    <div className="flex items-center gap-4 flex-wrap">
      <ToggleGroup
        type="single"
        value={activeTool}
        onValueChange={(value) => {
          if (value) onToolChange(value as "pen" | "eraser");
        }}
      >
        <ToggleGroupItem value="pen" aria-label="Pen tool">
          <Pen className="size-4" />
        </ToggleGroupItem>
        <ToggleGroupItem value="eraser" aria-label="Eraser tool">
          <Eraser className="size-4" />
        </ToggleGroupItem>
      </ToggleGroup>

      <Button variant="outline" size="icon" onClick={onClear} title="Clear canvas">
        <Trash2 className="size-4" />
      </Button>

      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground whitespace-nowrap">
          Stroke:
        </span>
        <Slider
          value={[strokeWidth]}
          onValueChange={([v]) => onStrokeWidthChange(v)}
          min={1}
          max={20}
          step={1}
          className="w-24"
        />
        <span className="text-sm text-muted-foreground w-6">{strokeWidth}</span>
      </div>
    </div>
  );
}
