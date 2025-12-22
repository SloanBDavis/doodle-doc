import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiGet, apiPost, apiPostFormData, apiDelete } from "./client";
import type {
  HealthResponse,
  Document,
  SearchResponse,
  IngestResponse,
  IngestStatusResponse,
} from "./types";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => apiGet<HealthResponse>("/health"),
    refetchInterval: 30000,
  });
}

export function useDocuments() {
  return useQuery({
    queryKey: ["documents"],
    queryFn: () => apiGet<Document[]>("/documents"),
  });
}

export function useSearch() {
  return useMutation({
    mutationFn: async (params: {
      sketchBlob: Blob;
      textQuery?: string;
      topK?: number;
      useRerank?: boolean;
    }) => {
      const formData = new FormData();
      formData.append("sketch_image", params.sketchBlob, "sketch.png");
      if (params.textQuery) {
        formData.append("text_query", params.textQuery);
      }
      formData.append("top_k", String(params.topK ?? 20));
      formData.append("use_rerank", String(params.useRerank ?? false));
      return apiPostFormData<SearchResponse>("/search", formData);
    },
  });
}

export function useStartIngest() {
  return useMutation({
    mutationFn: (params: { rootPath: string; forceReindex?: boolean }) =>
      apiPost<IngestResponse>("/ingest", {
        root_path: params.rootPath,
        force_reindex: params.forceReindex ?? false,
      }),
  });
}

export function useIngestStatus(jobId: string | null) {
  return useQuery({
    queryKey: ["ingest", jobId],
    queryFn: () => apiGet<IngestStatusResponse>(`/ingest/${jobId}`),
    enabled: !!jobId,
    refetchInterval: 1000,
  });
}

export function useRemoveDocuments() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (docIds: string[]) =>
      apiDelete<{ removed: number }>("/documents", { doc_ids: docIds }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["health"] });
    },
  });
}

export function useReindexDocuments() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (docIds: string[]) =>
      apiPost<{ reindexed: number }>("/documents/reindex", { doc_ids: docIds }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["health"] });
    },
  });
}
