interface ResultsSummaryProps {
  showing: number;
  queryTimeMs: number | null;
}

export function ResultsSummary({ showing, queryTimeMs }: ResultsSummaryProps) {
  if (showing === 0 && queryTimeMs === null) {
    return null;
  }

  return (
    <div className="flex items-center justify-between text-sm text-muted-foreground mb-4">
      <span>
        {showing > 0 ? `Showing ${showing} result${showing !== 1 ? "s" : ""}` : "No results"}
      </span>
      {queryTimeMs !== null && (
        <span>Query time: {queryTimeMs}ms</span>
      )}
    </div>
  );
}
