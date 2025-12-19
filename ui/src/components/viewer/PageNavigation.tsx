import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface PageNavigationProps {
  currentPage: number;
  totalPages: number;
  onPrevious: () => void;
  onNext: () => void;
}

export function PageNavigation({
  currentPage,
  totalPages,
  onPrevious,
  onNext,
}: PageNavigationProps) {
  return (
    <div className="flex items-center gap-2">
      <Button
        variant="outline"
        size="icon"
        onClick={onPrevious}
        disabled={currentPage <= 1}
      >
        <ChevronLeft className="size-4" />
      </Button>
      <span className="text-sm">
        Page {currentPage} of {totalPages}
      </span>
      <Button
        variant="outline"
        size="icon"
        onClick={onNext}
        disabled={currentPage >= totalPages}
      >
        <ChevronRight className="size-4" />
      </Button>
    </div>
  );
}
