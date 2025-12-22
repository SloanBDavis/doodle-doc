const BASE_URL = "/v1";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    throw new ApiError(res.status, `API error: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new ApiError(res.status, `API error: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function apiPostFormData<T>(
  path: string,
  formData: FormData,
): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new ApiError(res.status, `API error: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function apiDelete<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "DELETE",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    throw new ApiError(res.status, `API error: ${res.status}`);
  }
  return res.json() as Promise<T>;
}
