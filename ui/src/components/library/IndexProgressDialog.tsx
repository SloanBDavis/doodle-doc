import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import type { IngestStatusResponse } from "@/api/types";

interface IndexProgressDialogProps {
  isOpen: boolean;
  status: IngestStatusResponse | null;
  onClose: () => void;
}

export function IndexProgressDialog({
  isOpen,
  status,
  onClose,
}: IndexProgressDialogProps) {
  const progressPercent =
    status && status.pages_total > 0
      ? (status.pages_done / status.pages_total) * 100
      : 0;

  const isComplete = status?.status === "completed";

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open && isComplete) onClose();
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {isComplete ? "Indexing Complete" : "Indexing Documents"}
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <Progress value={progressPercent} />
          <div className="space-y-1 text-sm text-muted-foreground">
            <p>
              Documents: {status?.docs_done ?? 0} / {status?.docs_total ?? 0}
            </p>
            <p>
              Pages: {status?.pages_done ?? 0} / {status?.pages_total ?? 0}
            </p>
            {status?.eta_seconds != null && status.eta_seconds > 0 && (
              <p>Time remaining: ~{status.eta_seconds}s</p>
            )}
          </div>
          {isComplete && (
            <Button className="w-full" onClick={onClose}>
              Done
            </Button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
