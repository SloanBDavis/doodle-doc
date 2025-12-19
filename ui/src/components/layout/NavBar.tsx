import { Link } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { Search, Library } from "lucide-react";

export function NavBar() {
  return (
    <header className="border-b">
      <div className="container flex h-14 items-center">
        <Link to="/search" className="flex items-center gap-2 font-semibold">
          DoodleDoc
        </Link>
        <nav className="ml-auto flex gap-2">
          <Button variant="ghost" size="sm" asChild>
            <Link to="/search">
              <Search className="mr-2 size-4" />
              Search
            </Link>
          </Button>
          <Button variant="ghost" size="sm" asChild>
            <Link to="/library">
              <Library className="mr-2 size-4" />
              Library
            </Link>
          </Button>
        </nav>
      </div>
    </header>
  );
}
