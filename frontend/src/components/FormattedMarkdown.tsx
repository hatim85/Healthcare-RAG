import React from "react";

interface FormattedMarkdownProps {
  content: string;
  isStreaming?: boolean;
}

function renderInlineText(text: string): React.ReactNode[] {
  // Regex matches:
  // 1. Bold: **text**
  // 2. Citations: [S1], [S2], etc.
  // 3. Inline code: `text`
  // 4. Italic: *text*
  const pattern = /(\*\*[^*]+\*\*|\[S\d+\]|`[^`]+`|\*[^*]+\*)/g;
  const parts = text.split(pattern);

  return parts.map((part, idx) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length >= 4) {
      return (
        <strong key={idx} className="font-semibold text-slate-100">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length >= 2) {
      return (
        <em key={idx} className="italic text-slate-200">
          {part.slice(1, -1)}
        </em>
      );
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length >= 2) {
      return (
        <code
          key={idx}
          className="font-mono text-xs bg-slate-800 text-sky-300 px-1 py-0.5 rounded border border-slate-700"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    if (/^\[S\d+\]$/.test(part)) {
      return (
        <span
          key={idx}
          className="inline-block mx-0.5 px-1.5 py-0.2 font-mono text-[11px] font-semibold text-sky-400 bg-sky-500/15 border border-sky-500/30 rounded"
        >
          {part}
        </span>
      );
    }
    return part;
  });
}

export const FormattedMarkdown: React.FC<FormattedMarkdownProps> = ({
  content,
  isStreaming = false,
}) => {
  if (!content) return null;

  const lines = content.split("\n");
  const elements: React.ReactNode[] = [];
  let currentList: { type: "ul" | "ol"; items: string[] } | null = null;

  const flushList = () => {
    if (!currentList) return;
    const ListTag = currentList.type;
    const listKey = `list-${elements.length}`;
    elements.push(
      <ListTag
        key={listKey}
        className={
          currentList.type === "ul"
            ? "list-disc list-outside ml-5 space-y-1 my-2 text-slate-200"
            : "list-decimal list-outside ml-5 space-y-1 my-2 text-slate-200"
        }
      >
        {currentList.items.map((item, idx) => (
          <li key={idx} className="leading-relaxed">
            {renderInlineText(item)}
          </li>
        ))}
      </ListTag>
    );
    currentList = null;
  };

  lines.forEach((line, idx) => {
    const trimmed = line.trim();

    // Empty line
    if (!trimmed) {
      flushList();
      return;
    }

    // Unordered list item
    const ulMatch = line.match(/^\s*[-*•]\s+(.+)/);
    if (ulMatch) {
      if (!currentList || currentList.type !== "ul") {
        flushList();
        currentList = { type: "ul", items: [] };
      }
      currentList.items.push(ulMatch[1]);
      return;
    }

    // Ordered list item
    const olMatch = line.match(/^\s*\d+\.\s+(.+)/);
    if (olMatch) {
      if (!currentList || currentList.type !== "ol") {
        flushList();
        currentList = { type: "ol", items: [] };
      }
      currentList.items.push(olMatch[1]);
      return;
    }

    // Non-list line: flush any pending list
    flushList();

    // Headers
    if (trimmed.startsWith("### ")) {
      elements.push(
        <h4 key={idx} className="font-semibold text-slate-100 text-sm mt-3 mb-1">
          {renderInlineText(trimmed.slice(4))}
        </h4>
      );
    } else if (trimmed.startsWith("## ")) {
      elements.push(
        <h3 key={idx} className="font-bold text-slate-100 text-base mt-4 mb-1.5">
          {renderInlineText(trimmed.slice(3))}
        </h3>
      );
    } else if (trimmed.startsWith("# ")) {
      elements.push(
        <h2 key={idx} className="font-bold text-slate-100 text-lg mt-4 mb-2">
          {renderInlineText(trimmed.slice(2))}
        </h2>
      );
    } else {
      // Regular paragraph
      elements.push(
        <p key={idx} className="leading-relaxed my-1.5 text-slate-200">
          {renderInlineText(line)}
        </p>
      );
    }
  });

  flushList();

  return (
    <div className="space-y-0.5 text-slate-200 leading-relaxed font-sans">
      {elements}
      {isStreaming && (
        <span className="inline-block w-1.5 h-4 ml-1 bg-sky-400 animate-pulse align-middle" />
      )}
    </div>
  );
};
