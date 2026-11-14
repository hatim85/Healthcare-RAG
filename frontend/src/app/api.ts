import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import type {
  ProviderStatus,
  ToolInfo,
  ToolCallRequest,
  ToolCallResponse,
  MCPResource,
  MCPPrompt,
  LoginRequest,
  AuthToken,
  AskRequest,
  AskResponse,
  HealthResponse,
} from "@/types";

export const ragApi = createApi({
  reducerPath: "ragApi",
  baseQuery: fetchBaseQuery({
    baseUrl: "/",
    prepareHeaders: (headers) => {
      const token = localStorage.getItem("token");
      if (token) {
        headers.set("Authorization", `Bearer ${token}`);
      }
      return headers;
    },
  }),
  tagTypes: ["Providers", "Tools", "Resources", "Prompts", "Health"],
  endpoints: (builder) => ({
    getProviderStatus: builder.query<ProviderStatus[], void>({
      query: () => "api/mcp-providers/status",
      providesTags: ["Providers"],
    }),

    getAllTools: builder.query<ToolInfo[], void>({
      query: () => "api/mcp-providers/tools",
      providesTags: ["Tools"],
    }),

    getProviderTools: builder.query<ToolInfo[], string>({
      query: (provider) => `api/mcp-providers/tools/${provider}`,
      providesTags: ["Tools"],
    }),

    callTool: builder.mutation<ToolCallResponse, ToolCallRequest>({
      query: (body) => ({
        url: "api/mcp-providers/call",
        method: "POST",
        body,
      }),
    }),

    autoCallTool: builder.mutation<
      ToolCallResponse,
      { tool_name: string; arguments?: Record<string, unknown> }
    >({
      query: (body) => ({
        url: "api/mcp-providers/call/auto",
        method: "POST",
        body,
      }),
    }),

    getResources: builder.query<MCPResource[], void>({
      query: () => "api/mcp-providers/resources",
      providesTags: ["Resources"],
    }),

    getPrompts: builder.query<MCPPrompt[], void>({
      query: () => "api/mcp-providers/prompts",
      providesTags: ["Prompts"],
    }),

    login: builder.mutation<AuthToken, LoginRequest>({
      query: (credentials) => ({
        url: "auth/token",
        method: "POST",
        body: credentials,
      }),
    }),

    askQuestion: builder.mutation<AskResponse, AskRequest>({
      query: (body) => ({
        url: "api/ask",
        method: "POST",
        body,
      }),
    }),

    getHealth: builder.query<HealthResponse, void>({
      query: () => "api/health",
      providesTags: ["Health"],
    }),
  }),
});

export const {
  useGetProviderStatusQuery,
  useGetAllToolsQuery,
  useGetProviderToolsQuery,
  useCallToolMutation,
  useAutoCallToolMutation,
  useGetResourcesQuery,
  useGetPromptsQuery,
  useLoginMutation,
  useAskQuestionMutation,
  useGetHealthQuery,
} = ragApi;

