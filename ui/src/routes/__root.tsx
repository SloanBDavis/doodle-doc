import { createRootRoute, Outlet } from "@tanstack/react-router";
import { PageLayout } from "@/components/layout/PageLayout";

export const Route = createRootRoute({
  component: () => (
    <PageLayout>
      <Outlet />
    </PageLayout>
  ),
});
