import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";

interface AddFolderDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (path: string) => void;
  isLoading: boolean;
}

export function AddFolderDialog({
  isOpen,
  onClose,
  onSubmit,
  isLoading,
}: AddFolderDialogProps) {
  const [path, setPath] = useState("");

  const handleSubmit = () => {
    if (path.trim()) {
      onSubmit(path.trim());
    }
  };

  const handleClose = () => {
    setPath("");
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add Folder to Index</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="folder-path">Folder Path</Label>
            <Input
              id="folder-path"
              placeholder="/path/to/your/notes"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && path.trim()) {
                  handleSubmit();
                }
              }}
            />
            <p className="text-sm text-muted-foreground">
              Enter the absolute path to a folder containing PDF files.
            </p>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={handleClose}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={isLoading || !path.trim()}>
            {isLoading ? "Starting..." : "Index Folder"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
