import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import { EmptyState } from "@/components/shared/EmptyState";
import type { Document } from "@/api/types";
import { FileText } from "lucide-react";

interface DocumentTableProps {
  documents: Document[];
  selectedIds: Set<string>;
  onSelectionChange: (ids: Set<string>) => void;
}

export function DocumentTable({
  documents,
  selectedIds,
  onSelectionChange,
}: DocumentTableProps) {
  const toggleAll = () => {
    if (selectedIds.size === documents.length) {
      onSelectionChange(new Set());
    } else {
      onSelectionChange(new Set(documents.map((d) => d.doc_id)));
    }
  };

  const toggleOne = (docId: string) => {
    const newSet = new Set(selectedIds);
    if (newSet.has(docId)) {
      newSet.delete(docId);
    } else {
      newSet.add(docId);
    }
    onSelectionChange(newSet);
  };

  if (documents.length === 0) {
    return (
      <EmptyState
        icon={<FileText className="size-12" />}
        title="No documents indexed"
        description="Add a folder to start indexing your PDF notes"
      />
    );
  }

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-12">
              <Checkbox
                checked={
                  selectedIds.size === documents.length && documents.length > 0
                }
                onCheckedChange={toggleAll}
                aria-label="Select all"
              />
            </TableHead>
            <TableHead>Document</TableHead>
            <TableHead className="w-32">Pages</TableHead>
            <TableHead className="w-32">Status</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {documents.map((doc) => (
            <TableRow key={doc.doc_id}>
              <TableCell>
                <Checkbox
                  checked={selectedIds.has(doc.doc_id)}
                  onCheckedChange={() => toggleOne(doc.doc_id)}
                  aria-label={`Select ${doc.name}`}
                />
              </TableCell>
              <TableCell className="font-medium" title={doc.path}>
                {doc.name}
              </TableCell>
              <TableCell>{doc.num_pages} pages</TableCell>
              <TableCell>
                <span className="inline-flex items-center rounded-full bg-green-100 px-2 py-1 text-xs font-medium text-green-700">
                  Indexed
                </span>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
