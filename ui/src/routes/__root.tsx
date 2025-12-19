import { createRootRoute, Outlet } from "@tanstack/react-router";
import { PageLayout } from "@/components/layout/PageLayout";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";

export const Route = createRootRoute({
  component: () => (
    <ErrorBoundary>
      <PageLayout>
        <Outlet />
      </PageLayout>
    </ErrorBoundary>
  ),
});
