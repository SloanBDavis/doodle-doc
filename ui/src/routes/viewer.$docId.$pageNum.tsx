import { useState } from "react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { PageImage } from "@/components/viewer/PageImage";
import { ZoomControls } from "@/components/viewer/ZoomControls";
import { PageNavigation } from "@/components/viewer/PageNavigation";
import { useDocuments } from "@/api/hooks";

export const Route = createFileRoute("/viewer/$docId/$pageNum")({
  component: ViewerPage,
});

function ViewerPage() {
  const navigate = useNavigate();
  const { docId, pageNum } = Route.useParams();
  const currentPage = parseInt(pageNum, 10);

  const [zoom, setZoom] = useState(1);
  const [fitMode, setFitMode] = useState<"width" | "page" | "none">("width");

  const { data: documents } = useDocuments();
  const doc = documents?.find((d) => d.doc_id === docId);
  const totalPages = doc?.num_pages ?? 1;

  const imageSrc = `/v1/doc/${docId}/page/${currentPage}`;

  const goToPage = (page: number) => {
    navigate({
      to: "/viewer/$docId/$pageNum",
      params: { docId, pageNum: String(page) },
    });
  };

  return (
    <div className="h-[calc(100vh-3.5rem)] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-4">
          <Button variant="ghost" onClick={() => navigate({ to: "/search" })}>
            <ArrowLeft className="mr-2 size-4" />
            Back to Results
          </Button>
          <span className="font-medium">{doc?.name ?? "Loading..."}</span>
        </div>
        <PageNavigation
          currentPage={currentPage}
          totalPages={totalPages}
          onPrevious={() => goToPage(currentPage - 1)}
          onNext={() => goToPage(currentPage + 1)}
        />
      </div>

      {/* Main content */}
      <PageImage src={imageSrc} zoom={zoom} fitMode={fitMode} />

      {/* Footer with zoom controls */}
      <div className="flex justify-center p-4 border-t">
        <ZoomControls
          zoom={zoom}
          fitMode={fitMode}
          onZoomChange={setZoom}
          onFitModeChange={setFitMode}
        />
      </div>
    </div>
  );
}
