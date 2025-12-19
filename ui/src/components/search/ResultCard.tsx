import { Card, CardContent } from "@/components/ui/card";
import type { SearchResultItem } from "@/api/types";

interface ResultCardProps {
  result: SearchResultItem;
  onClick: () => void;
}

export function ResultCard({ result, onClick }: ResultCardProps) {
  return (
    <Card
      className="cursor-pointer hover:ring-2 hover:ring-primary transition-all overflow-hidden"
      onClick={onClick}
    >
      <CardContent className="p-2">
        <img
          src={result.thumbnail_url}
          alt={`${result.doc_name} page ${result.page_num}`}
          className="w-full aspect-[3/4] object-cover rounded"
          loading="lazy"
        />
        <div className="mt-2 space-y-0.5">
          <p className="text-sm font-medium truncate" title={result.doc_name}>
            {result.doc_name}
          </p>
          <p className="text-xs text-muted-foreground">Page {result.page_num}</p>
          <p className="text-xs text-muted-foreground">
            {(result.score * 100).toFixed(0)}% match
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
