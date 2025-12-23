export interface SearchResultItem {
  doc_id: string;
  doc_name: string;
  page_num: number;
  score: number;
  stage: "fast" | "reranked" | "colqwen2";
  thumbnail_url: string;
}

export interface SearchResponse {
  results: SearchResultItem[];
  query_time_ms: number;
  total_indexed_pages: number;
}

export interface Document {
  doc_id: string;
  path: string;
  name: string;
  num_pages: number;
  sha256: string;
}

export interface IngestRequest {
  root_path: string;
  force_reindex?: boolean;
}

export interface IngestResponse {
  job_id: string;
  status: string;
}

export interface IngestStatusResponse {
  status: string;
  docs_done: number;
  docs_total: number;
  pages_done: number;
  pages_total: number;
  eta_seconds: number | null;
}

export interface HealthResponse {
  status: string;
  siglip_loaded: boolean;
  colqwen_loaded: boolean;
  indexed_pages: number;
  index_size_mb: number;
}

export interface ModelLoadResponse {
  status: string;
  message: string;
}
