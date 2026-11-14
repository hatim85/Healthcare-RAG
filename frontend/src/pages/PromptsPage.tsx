import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  MessageSquare,
  Copy,
  Check,
  Stethoscope,
  Sparkles,
} from "lucide-react";
import { useGetPromptsQuery } from "@/app/api";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorAlert } from "@/components/ErrorAlert";

export const PromptsPage: React.FC = () => {
  const navigate = useNavigate();
  const { data: prompts, isLoading, error } = useGetPromptsQuery();
  const [copiedName, setCopiedName] = useState<string | null>(null);

  const handleCopy = (text: string, name: string) => {
    navigator.clipboard.writeText(text);
    setCopiedName(name);
    setTimeout(() => setCopiedName(null), 2000);
  };

  const handleUseInChat = (promptName: string) => {
    navigate(`/clinical?prompt=${encodeURIComponent(promptName)}`);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <MessageSquare className="w-8 h-8 text-sky-400" />
            MCP Prompts Catalog
          </h2>
          <p className="text-slate-400 text-sm mt-1">
            Prompt templates provided by Langfuse, Clinical RAG, and external providers
          </p>
        </div>

        <div className="text-xs text-slate-400 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-xl self-start sm:self-auto font-mono">
          {prompts?.length || 0} Registered Prompts
        </div>
      </div>

      {error && (
        <ErrorAlert message="Failed to load prompts from connected MCP servers." />
      )}

      {isLoading ? (
        <LoadingSpinner message="Fetching prompts from MCP providers..." />
      ) : !prompts || prompts.length === 0 ? (
        <div className="text-center py-16 bg-slate-900/60 border border-dashed border-slate-800 rounded-2xl">
          <Sparkles className="w-10 h-10 text-slate-600 mx-auto mb-3" />
          <h3 className="text-slate-300 font-semibold text-sm">
            No MCP prompts registered
          </h3>
          <p className="text-slate-500 text-xs mt-1">
            Ensure Langfuse MCP or Healthcare RAG prompt templates are configured.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {prompts.map((p) => {
            const isCopied = copiedName === p.name;
            return (
              <div
                key={`${p.provider}-${p.name}`}
                className="p-5 bg-slate-900/90 border border-slate-800 hover:border-slate-700 rounded-2xl transition-all shadow-md flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <span className="text-xs font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-sky-500/15 text-sky-400 border border-sky-500/30">
                      {p.provider}
                    </span>
                    <button
                      onClick={() => handleCopy(p.name, p.name)}
                      className="text-xs text-slate-400 hover:text-white flex items-center gap-1 cursor-pointer"
                      title="Copy prompt identifier"
                    >
                      {isCopied ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                          <span className="text-emerald-400">Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span>Copy ID</span>
                        </>
                      )}
                    </button>
                  </div>

                  <h3 className="text-base font-bold text-white font-mono mb-2">
                    {p.name}
                  </h3>

                  <p className="text-xs text-slate-400 leading-relaxed">
                    {p.description || "Structured clinical prompt template."}
                  </p>
                </div>

                <div className="mt-5 pt-3 border-t border-slate-800/80 flex items-center justify-end">
                  <button
                    onClick={() => handleUseInChat(p.name)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-colors cursor-pointer"
                  >
                    <Stethoscope className="w-3 h-3 text-sky-400" />
                    Load in Clinical Chat
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

