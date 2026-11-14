import React, { useState } from "react";
import {
  FolderOpen,
  Copy,
  Check,
  ExternalLink,
  Layers,
} from "lucide-react";
import { useGetResourcesQuery } from "@/app/api";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorAlert } from "@/components/ErrorAlert";

export const ResourcesPage: React.FC = () => {
  const { data: resources, isLoading, error } = useGetResourcesQuery();
  const [copiedUri, setCopiedUri] = useState<string | null>(null);

  const handleCopy = (uri: string) => {
    navigator.clipboard.writeText(uri);
    setCopiedUri(uri);
    setTimeout(() => setCopiedUri(null), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <FolderOpen className="w-8 h-8 text-sky-400" />
            MCP Resources
          </h2>
          <p className="text-slate-400 text-sm mt-1">
            Data feeds, catalogs, and state endpoints exposed across MCP servers
          </p>
        </div>

        <div className="text-xs text-slate-400 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-xl self-start sm:self-auto font-mono">
          {resources?.length || 0} Registered Endpoints
        </div>
      </div>

      {error && (
        <ErrorAlert message="Failed to load resources from connected MCP servers." />
      )}

      {isLoading ? (
        <LoadingSpinner message="Scanning MCP resources from connected servers..." />
      ) : !resources || resources.length === 0 ? (
        <div className="text-center py-16 bg-slate-900/60 border border-dashed border-slate-800 rounded-2xl">
          <Layers className="w-10 h-10 text-slate-600 mx-auto mb-3" />
          <h3 className="text-slate-300 font-semibold text-sm">
            No MCP resources exposed
          </h3>
          <p className="text-slate-500 text-xs mt-1">
            Ensure your MCP servers have registered resources (e.g. guidelines://catalog).
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {resources.map((r) => {
            const isCopied = copiedUri === r.uri;
            return (
              <div
                key={`${r.provider}-${r.uri}`}
                className="p-5 bg-slate-900/90 border border-slate-800 hover:border-slate-700 rounded-2xl transition-all shadow-md flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <span className="text-xs font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-sky-500/15 text-sky-400 border border-sky-500/30">
                      {r.provider}
                    </span>
                    <button
                      onClick={() => handleCopy(r.uri)}
                      className="text-xs text-slate-400 hover:text-white flex items-center gap-1 cursor-pointer"
                      title="Copy URI"
                    >
                      {isCopied ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                          <span className="text-emerald-400">Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span>Copy URI</span>
                        </>
                      )}
                    </button>
                  </div>

                  <h3 className="text-sm font-bold text-sky-300 font-mono mb-2 break-all flex items-center gap-1.5">
                    <ExternalLink className="w-3.5 h-3.5 shrink-0 text-slate-500" />
                    {r.uri}
                  </h3>

                  <div className="font-semibold text-slate-200 text-xs mb-1">
                    {r.name || "Resource"}
                  </div>

                  <p className="text-xs text-slate-400 leading-relaxed">
                    {r.description || "Read-only resource accessible via Model Context Protocol."}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

