import React from "react";
import { Wrench, Play, ShieldCheck } from "lucide-react";

interface ToolCardProps {
  name: string;
  description: string;
  provider: string;
  onExecute: () => void;
}

export const ToolCard: React.FC<ToolCardProps> = ({
  name,
  description,
  provider,
  onExecute,
}) => {
  const getProviderColor = (p: string) => {
    switch (p.toLowerCase()) {
      case "langfuse":
        return "bg-amber-500/15 text-amber-300 border-amber-500/30";
      case "qdrant":
        return "bg-rose-500/15 text-rose-300 border-rose-500/30";
      case "groq":
        return "bg-orange-500/15 text-orange-300 border-orange-500/30";
      case "firecrawl":
        return "bg-emerald-500/15 text-emerald-300 border-emerald-500/30";
      case "litellm":
        return "bg-indigo-500/15 text-indigo-300 border-indigo-500/30";
      case "redis":
        return "bg-red-500/15 text-red-300 border-red-500/30";
      default:
        return "bg-sky-500/15 text-sky-300 border-sky-500/30";
    }
  };

  return (
    <div className="flex flex-col justify-between p-5 bg-slate-800/70 hover:bg-slate-800 border border-slate-700/80 hover:border-sky-500/50 rounded-2xl transition-all duration-200 group shadow-lg shadow-black/20">
      <div>
        <div className="flex items-center justify-between gap-2 mb-3">
          <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider border ${getProviderColor(
              provider
            )}`}
          >
            {provider}
          </span>
          <Wrench className="w-4 h-4 text-slate-500 group-hover:text-sky-400 transition-colors" />
        </div>

        <h3 className="font-semibold text-slate-100 text-base mb-2 font-mono flex items-center gap-1.5 break-all">
          {name}
        </h3>

        <p className="text-slate-400 text-xs leading-relaxed line-clamp-3 mb-4">
          {description || "No tool description provided."}
        </p>
      </div>

      <div className="pt-3 border-t border-slate-700/60 flex items-center justify-between gap-3">
        <span className="inline-flex items-center gap-1 text-[11px] text-slate-400">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          MCP Verified
        </span>

        <button
          onClick={onExecute}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-xs font-medium transition-all shadow-md shadow-sky-950/40 cursor-pointer"
        >
          <Play className="w-3 h-3 fill-white" />
          Execute
        </button>
      </div>
    </div>
  );
};

