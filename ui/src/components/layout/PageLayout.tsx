import type { ReactNode } from "react";
import { NavBar } from "./NavBar";
import { Toaster } from "@/components/ui/sonner";

interface PageLayoutProps {
  children: ReactNode;
}

export function PageLayout({ children }: PageLayoutProps) {
  return (
    <div className="min-h-screen flex flex-col">
      <NavBar />
      <main className="flex-1">{children}</main>
      <Toaster />
    </div>
  );
}
