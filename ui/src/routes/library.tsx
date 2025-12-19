import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/library")({
  component: LibraryPage,
});

function LibraryPage() {
  return (
    <div className="container py-6">
      <h1 className="text-2xl font-bold">Document Library</h1>
      <p className="text-muted-foreground">Library page will be implemented here.</p>
    </div>
  );
}
