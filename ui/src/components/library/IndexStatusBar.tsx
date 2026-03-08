import { Card, CardContent } from "@/components/ui/card";
import { CheckCircle, Circle } from "lucide-react";

interface IndexStatusBarProps {
  totalPages: number;
  indexSizeMb: number;
  colqwenLoaded: boolean;
}

export function IndexStatusBar({
  totalPages,
  indexSizeMb,
  colqwenLoaded,
}: IndexStatusBarProps) {
  return (
    <Card>
      <CardContent className="py-4">
        <div className="flex flex-wrap justify-between gap-4 text-sm">
          <span>
            Index Status: {totalPages} pages indexed, {indexSizeMb.toFixed(1)} MB
          </span>
          <div className="flex gap-4">
            <span className="flex items-center gap-1">
              {colqwenLoaded ? (
                <CheckCircle className="size-4 text-green-500" />
              ) : (
                <Circle className="size-4 text-muted-foreground" />
              )}
              ColQwen2
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
