import React, { useState } from "react";
import { FileText, ChevronDown, ChevronUp } from "lucide-react";

interface SourceCardProps {
  sourceId: string;
  filename: string;
  page: string | number;
  text: string;
}

export const SourceCard: React.FC<SourceCardProps> = ({
  sourceId,
  filename,
  page,
  text,
}) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-slate-800/90 border border-slate-700/80 rounded-xl p-3.5 text-xs text-slate-300 transition-colors hover:border-slate-600">
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <span className="font-mono font-bold px-2 py-0.5 bg-sky-500/20 text-sky-400 border border-sky-500/30 rounded-md">
            [{sourceId}]
          </span>
          <span className="flex items-center gap-1 font-medium text-slate-200 truncate max-w-[200px] sm:max-w-[320px]">
            <FileText className="w-3.5 h-3.5 text-slate-400 shrink-0" />
            {filename}
          </span>
        </div>
        <span className="text-[11px] text-slate-400 bg-slate-900/60 px-2 py-0.5 rounded">
          Page {page}
        </span>
      </div>

      <div
        className={`text-slate-300 leading-relaxed font-sans ${
          expanded ? "whitespace-pre-wrap" : "line-clamp-2"
        }`}
      >
        "{text}"
      </div>

      {text.length > 140 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 inline-flex items-center gap-1 text-[11px] text-sky-400 hover:text-sky-300 font-medium cursor-pointer"
        >
          {expanded ? (
            <>
              Show less <ChevronUp className="w-3 h-3" />
            </>
          ) : (
            <>
              Show full excerpt <ChevronDown className="w-3 h-3" />
            </>
          )}
        </button>
      )}
    </div>
  );
};

