import { useState, useRef } from "react";
import { useNavigate, createFileRoute } from "@tanstack/react-router";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  SketchCanvas,
  type SketchCanvasRef,
} from "@/components/search/SketchCanvas";
import { ToolBar } from "@/components/search/ToolBar";
import { ResultsGrid } from "@/components/search/ResultsGrid";
import { ResultsSummary } from "@/components/search/ResultsSummary";
import { useSearch } from "@/api/hooks";
import type { SearchResultItem } from "@/api/types";

export const Route = createFileRoute("/search")({
  component: SearchPage,
});

function SearchPage() {
  const navigate = useNavigate();
  const canvasRef = useRef<SketchCanvasRef>(null);

  const [tool, setTool] = useState<"pen" | "eraser">("pen");
  const [strokeWidth, setStrokeWidth] = useState(4);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [queryTimeMs, setQueryTimeMs] = useState<number | null>(null);

  const searchMutation = useSearch();

  const handleSearch = async () => {
    const blob = await canvasRef.current?.exportBlob();
    if (!blob) return;

    try {
      const response = await searchMutation.mutateAsync({
        sketchBlob: blob,
      });

      setResults(response.results);
      setQueryTimeMs(response.query_time_ms);
    } catch {
      toast.error("Search failed. Make sure the API server is running.");
    }
  };

  const handleResultClick = (result: SearchResultItem) => {
    navigate({
      to: "/viewer/$docId/$pageNum",
      params: { docId: result.doc_id, pageNum: String(result.page_num) },
    });
  };

  return (
    <div className="container py-6">
      <div className="grid grid-cols-1 lg:grid-cols-[400px_1fr] gap-8">
        <div className="space-y-4">
          <SketchCanvas
            ref={canvasRef}
            width={400}
            height={400}
            strokeWidth={strokeWidth}
            tool={tool}
          />
          <ToolBar
            activeTool={tool}
            strokeWidth={strokeWidth}
            onToolChange={setTool}
            onStrokeWidthChange={setStrokeWidth}
            onClear={() => canvasRef.current?.clear()}
          />
          <Button
            className="w-full"
            onClick={handleSearch}
            disabled={searchMutation.isPending}
          >
            {searchMutation.isPending ? "Searching..." : "Search"}
          </Button>
        </div>

        <div>
          <ResultsSummary showing={results.length} queryTimeMs={queryTimeMs} />
          <ResultsGrid results={results} onResultClick={handleResultClick} />
        </div>
      </div>
    </div>
  );
}
