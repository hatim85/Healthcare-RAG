import React, { useState, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Wrench,
  Search,
  Play,
  X,
  CheckCircle2,
  AlertCircle,
  Copy,
  Check,
  Code,
  Terminal,
} from "lucide-react";
import { useGetAllToolsQuery, useCallToolMutation } from "@/app/api";
import { ToolCard } from "@/components/ToolCard";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorAlert } from "@/components/ErrorAlert";
import type { ToolInfo } from "@/types";

export const ToolsPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedProviderParam = searchParams.get("provider") || "all";

  const { data: tools, isLoading, error } = useGetAllToolsQuery();
  const [callTool, { isLoading: isCalling }] = useCallToolMutation();

  const [searchQuery, setSearchQuery] = useState("");
  const [activeTool, setActiveTool] = useState<ToolInfo | null>(null);
  const [argumentsJson, setArgumentsJson] = useState("{}");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [callResult, setCallResult] = useState<{
    is_error: boolean;
    content: string;
  } | null>(null);
  const [copied, setCopied] = useState(false);

  // Extract unique providers for filter tabs
  const providersList = useMemo(() => {
    if (!tools) return [];
    const set = new Set<string>();
    tools.forEach((t) => {
      if (t.provider) set.add(t.provider);
    });
    return Array.from(set).sort();
  }, [tools]);

  // Filter tools by tab and search input
  const filteredTools = useMemo(() => {
    if (!tools) return [];
    return tools.filter((tool) => {
      const matchProvider =
        selectedProviderParam === "all" ||
        tool.provider?.toLowerCase() === selectedProviderParam.toLowerCase();
      const query = searchQuery.toLowerCase().trim();
      const matchSearch =
        !query ||
        tool.name.toLowerCase().includes(query) ||
        tool.description?.toLowerCase().includes(query) ||
        tool.provider?.toLowerCase().includes(query);
      return matchProvider && matchSearch;
    });
  }, [tools, selectedProviderParam, searchQuery]);

  const handleOpenExecute = (tool: ToolInfo) => {
    setActiveTool(tool);
    setCallResult(null);
    setJsonError(null);

    // Provide friendly initial template arguments based on common tool names
    let defaultArgs = "{}";
    if (tool.name === "getPrompt") {
      defaultArgs = JSON.stringify({ name: "clinical_consultation" }, null, 2);
    } else if (tool.name === "listPrompts") {
      defaultArgs = JSON.stringify({ page: 1, limit: 10 }, null, 2);
    } else if (tool.name.includes("find")) {
      defaultArgs = JSON.stringify(
        { query: "diabetes guidelines first-line treatment" },
        null,
        2
      );
    } else if (tool.name.includes("store")) {
      defaultArgs = JSON.stringify(
        { information: "Patient diagnosed with hypertension." },
        null,
        2
      );
    } else if (tool.name.includes("scrape")) {
      defaultArgs = JSON.stringify(
        { url: "https://example.com" },
        null,
        2
      );
    }
    setArgumentsJson(defaultArgs);
  };

  const handleExecute = async () => {
    if (!activeTool) return;
    setJsonError(null);
    setCallResult(null);

    let parsedArgs = {};
    try {
      parsedArgs = JSON.parse(argumentsJson || "{}");
    } catch (e: any) {
      setJsonError(`Invalid JSON format: ${e.message}`);
      return;
    }

    try {
      const res = await callTool({
        provider: activeTool.provider,
        tool_name: activeTool.name,
        arguments: parsedArgs,
      }).unwrap();

      setCallResult({
        is_error: res.is_error,
        content: res.content,
      });
    } catch (err: any) {
      setCallResult({
        is_error: true,
        content: err?.data?.detail || err?.message || "Tool execution failed",
      });
    }
  };

  const handleCopyResult = () => {
    if (callResult?.content) {
      navigator.clipboard.writeText(callResult.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Title & Search */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <Wrench className="w-8 h-8 text-sky-400" />
            MCP Tools Explorer
          </h2>
          <p className="text-slate-400 text-sm mt-1">
            Discover and invoke capabilities across connected MCP providers
          </p>
        </div>

        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search tools or providers..."
            className="w-full bg-slate-900 border border-slate-800 focus:border-sky-500 rounded-xl pl-10 pr-4 py-2 text-xs text-white placeholder-slate-500 outline-none transition-colors"
          />
        </div>
      </div>

      {error && (
        <ErrorAlert message="Failed to load tool definitions from external MCP providers." />
      )}

      {/* Provider Filter Tabs */}
      <div className="flex items-center gap-2 overflow-x-auto pb-2 scrollbar-none">
        <button
          onClick={() => setSearchParams({})}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors whitespace-nowrap cursor-pointer ${
            selectedProviderParam === "all"
              ? "bg-sky-600 text-white shadow-md shadow-sky-950"
              : "bg-slate-900 text-slate-400 hover:text-white hover:bg-slate-800 border border-slate-800"
          }`}
        >
          All Providers ({tools?.length || 0})
        </button>

        {providersList.map((provider) => {
          const count = tools?.filter((t) => t.provider === provider).length || 0;
          const isSelected =
            selectedProviderParam.toLowerCase() === provider.toLowerCase();
          return (
            <button
              key={provider}
              onClick={() => setSearchParams({ provider })}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors whitespace-nowrap cursor-pointer ${
                isSelected
                  ? "bg-sky-600 text-white shadow-md shadow-sky-950"
                  : "bg-slate-900 text-slate-400 hover:text-white hover:bg-slate-800 border border-slate-800"
              }`}
            >
              {provider} ({count})
            </button>
          );
        })}
      </div>

      {/* Tools Grid */}
      {isLoading ? (
        <LoadingSpinner message="Scanning tools from connected MCP servers..." />
      ) : filteredTools.length === 0 ? (
        <div className="text-center py-16 bg-slate-900/60 border border-dashed border-slate-800 rounded-2xl">
          <Wrench className="w-10 h-10 text-slate-600 mx-auto mb-3" />
          <h3 className="text-slate-300 font-semibold text-sm">
            No MCP tools found
          </h3>
          <p className="text-slate-500 text-xs mt-1">
            Try adjusting your search criteria or make sure providers are connected in .env.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredTools.map((tool) => (
            <ToolCard
              key={`${tool.provider}-${tool.name}`}
              name={tool.name}
              description={tool.description}
              provider={tool.provider}
              onExecute={() => handleOpenExecute(tool)}
            />
          ))}
        </div>
      )}

      {/* Execution Modal */}
      {activeTool && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-2xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-5 border-b border-slate-800 bg-slate-950/60">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-sky-500/15 border border-sky-500/30 text-sky-400">
                  <Terminal className="w-5 h-5" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-bold uppercase text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded border border-sky-500/20">
                      {activeTool.provider}
                    </span>
                    <h3 className="text-base font-bold text-white font-mono">
                      {activeTool.name}
                    </h3>
                  </div>
                  <p className="text-xs text-slate-400 mt-0.5 line-clamp-1">
                    {activeTool.description || "Execute tool over Model Context Protocol"}
                  </p>
                </div>
              </div>

              <button
                onClick={() => setActiveTool(null)}
                className="text-slate-400 hover:text-white p-1 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-5 overflow-y-auto space-y-4 flex-1">
              <div>
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 flex items-center justify-between">
                  <span>Input Arguments (JSON format)</span>
                  <Code className="w-3.5 h-3.5 text-slate-500" />
                </label>
                <textarea
                  value={argumentsJson}
                  onChange={(e) => setArgumentsJson(e.target.value)}
                  rows={5}
                  className="w-full bg-slate-950 border border-slate-800 focus:border-sky-500 rounded-xl p-3 font-mono text-xs text-slate-200 outline-none leading-relaxed transition-colors resize-y"
                  placeholder='{\n  "param": "value"\n}'
                />
                {jsonError && (
                  <p className="text-xs text-red-400 mt-1 flex items-center gap-1">
                    <AlertCircle className="w-3.5 h-3.5" />
                    {jsonError}
                  </p>
                )}
              </div>

              {/* Result Area */}
              {callResult && (
                <div className="space-y-2 pt-2 border-t border-slate-800">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5 text-xs font-semibold">
                      {callResult.is_error ? (
                        <>
                          <AlertCircle className="w-4 h-4 text-red-400" />
                          <span className="text-red-400">Execution Error</span>
                        </>
                      ) : (
                        <>
                          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                          <span className="text-emerald-400">Execution Success</span>
                        </>
                      )}
                    </div>

                    <button
                      onClick={handleCopyResult}
                      className="text-xs text-slate-400 hover:text-white flex items-center gap-1 px-2 py-1 rounded bg-slate-800"
                    >
                      {copied ? (
                        <>
                          <Check className="w-3 h-3 text-emerald-400" /> Copied
                        </>
                      ) : (
                        <>
                          <Copy className="w-3 h-3" /> Copy
                        </>
                      )}
                    </button>
                  </div>

                  <pre className="p-3 bg-slate-950 border border-slate-800 rounded-xl text-xs font-mono text-slate-200 overflow-x-auto max-h-60 leading-relaxed whitespace-pre-wrap">
                    {callResult.content}
                  </pre>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-slate-800 bg-slate-950/60 flex items-center justify-end gap-3">
              <button
                onClick={() => setActiveTool(null)}
                className="px-4 py-2 text-xs font-medium text-slate-400 hover:text-white rounded-xl transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleExecute}
                disabled={isCalling}
                className="inline-flex items-center gap-2 px-5 py-2 bg-sky-600 hover:bg-sky-500 disabled:bg-sky-900 text-white text-xs font-semibold rounded-xl transition-all shadow-lg shadow-sky-900/40 cursor-pointer"
              >
                {isCalling ? (
                  <>Running...</>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-white" />
                    Invoke Tool
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
