import { ResultCard } from "./ResultCard";
import { EmptyState } from "@/components/shared/EmptyState";
import type { SearchResultItem } from "@/api/types";
import { Search } from "lucide-react";

interface ResultsGridProps {
  results: SearchResultItem[];
  onResultClick: (result: SearchResultItem) => void;
}

export function ResultsGrid({ results, onResultClick }: ResultsGridProps) {
  if (results.length === 0) {
    return (
      <EmptyState
        icon={<Search className="size-12" />}
        title="No results yet"
        description="Draw a sketch and click Search to find matching pages"
      />
    );
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
      {results.map((result, index) => (
        <ResultCard
          key={`${result.doc_id}-${result.page_num}-${index}`}
          result={result}
          onClick={() => onResultClick(result)}
        />
      ))}
    </div>
  );
}
