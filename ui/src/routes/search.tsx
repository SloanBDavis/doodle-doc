import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/search")({
  component: SearchPage,
});

function SearchPage() {
  return (
    <div className="container py-6">
      <h1 className="text-2xl font-bold">Search</h1>
      <p className="text-muted-foreground">Search page will be implemented here.</p>
    </div>
  );
}
