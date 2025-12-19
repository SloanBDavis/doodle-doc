import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiGet, apiPost, apiPostFormData } from "./client";
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
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (params: { rootPath: string; forceReindex?: boolean }) =>
      apiPost<IngestResponse>("/ingest", {
        root_path: params.rootPath,
        force_reindex: params.forceReindex ?? false,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
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
