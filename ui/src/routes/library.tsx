import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { FolderPlus } from "lucide-react";
import { DocumentTable } from "@/components/library/DocumentTable";
import { IndexStatusBar } from "@/components/library/IndexStatusBar";
import { AddFolderDialog } from "@/components/library/AddFolderDialog";
import { IndexProgressDialog } from "@/components/library/IndexProgressDialog";
import { LoadingSpinner } from "@/components/shared/LoadingSpinner";
import {
  useDocuments,
  useHealth,
  useStartIngest,
  useIngestStatus,
} from "@/api/hooks";

export const Route = createFileRoute("/library")({
  component: LibraryPage,
});

function LibraryPage() {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showAddFolder, setShowAddFolder] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);

  const { data: documents = [], isLoading: documentsLoading } = useDocuments();
  const { data: health } = useHealth();
  const startIngest = useStartIngest();
  const { data: ingestStatus } = useIngestStatus(currentJobId);

  const handleStartIndex = async (path: string) => {
    const response = await startIngest.mutateAsync({ rootPath: path });
    setCurrentJobId(response.job_id);
    setShowAddFolder(false);
  };

  const handleProgressClose = () => {
    setCurrentJobId(null);
  };

  if (documentsLoading) {
    return (
      <div className="container py-6 flex justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="container py-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Document Library</h1>
        <Button onClick={() => setShowAddFolder(true)}>
          <FolderPlus className="mr-2 size-4" />
          Add Folder
        </Button>
      </div>

      <DocumentTable
        documents={documents}
        selectedIds={selectedIds}
        onSelectionChange={setSelectedIds}
      />

      {selectedIds.size > 0 && (
        <div className="flex items-center gap-4">
          <span className="text-sm text-muted-foreground">
            {selectedIds.size} selected
          </span>
          <Button variant="outline" size="sm">
            Re-index Selected
          </Button>
          <Button variant="outline" size="sm">
            Remove Selected
          </Button>
        </div>
      )}

      <IndexStatusBar
        totalPages={health?.indexed_pages ?? 0}
        indexSizeMb={health?.index_size_mb ?? 0}
        siglipLoaded={health?.siglip_loaded ?? false}
        colqwenLoaded={health?.colqwen_loaded ?? false}
      />

      <AddFolderDialog
        isOpen={showAddFolder}
        onClose={() => setShowAddFolder(false)}
        onSubmit={handleStartIndex}
        isLoading={startIngest.isPending}
      />

      <IndexProgressDialog
        isOpen={currentJobId !== null}
        status={ingestStatus ?? null}
        onClose={handleProgressClose}
      />
    </div>
  );
}
