// ─── MCP Provider Types ─────────────────────────────────────────────

export interface ProviderStatus {
  name: string;
  transport: string;
  enabled: boolean;
  connected: boolean;
  error: string | null;
  tools_count: number;
}

export interface ToolInfo {
  name: string;
  description: string;
  provider: string;
}

export interface ToolCallRequest {
  provider: string;
  tool_name: string;
  arguments: Record<string, unknown>;
}

export interface ToolCallResponse {
  provider: string;
  tool_name: string;
  is_error: boolean;
  content: string;
}

export interface MCPResource {
  uri: string;
  name: string;
  description: string;
  provider: string;
}

export interface MCPPrompt {
  name: string;
  description: string;
  provider: string;
}

// ─── Auth Types ─────────────────────────────────────────────────────

export interface LoginRequest {
  username: string;
  password: string;
  tenant_id?: string;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
  expires_in: number;
  role: string;
  username: string;
  tenant_id: string;
}

// ─── Clinical / RAG Types ───────────────────────────────────────────

export interface AskRequest {
  question: string;
  session_id?: string;
  verbose?: boolean;
  stream?: boolean;
}

export interface AskResponse {
  status: string;
  answer: string;
  question: string;
  sources: Source[];
  citation_check?: Record<string, unknown>;
  claim_verification?: Record<string, unknown>;
  model?: string;
  observability?: Record<string, unknown>;
}

export interface Source {
  source_id: string;
  filename: string;
  page: number | string;
  text: string;
}

// ─── Health Check ───────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  qdrant: string;
  sqlite_registry: string;
  database: string;
  redis: string;
  active_jobs: number;
  timestamp: string;
}
