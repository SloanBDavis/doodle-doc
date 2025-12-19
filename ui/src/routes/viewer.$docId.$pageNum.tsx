import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/viewer/$docId/$pageNum")({
  component: ViewerPage,
});

function ViewerPage() {
  const { docId, pageNum } = Route.useParams();

  return (
    <div className="container py-6">
      <h1 className="text-2xl font-bold">Page Viewer</h1>
      <p className="text-muted-foreground">
        Viewing document {docId}, page {pageNum}
      </p>
    </div>
  );
}
