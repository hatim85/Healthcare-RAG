import React from "react";

interface StatusBadgeProps {
  connected: boolean;
  label?: string;
  size?: "sm" | "md";
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  connected,
  label,
  size = "md",
}) => {
  const isSm = size === "sm";

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-medium rounded-full ${
        isSm ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-xs"
      } ${
        connected
          ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
          : "bg-red-500/15 text-red-400 border border-red-500/30"
      }`}
    >
      <span
        className={`rounded-full shrink-0 ${isSm ? "w-1.5 h-1.5" : "w-2 h-2"} ${
          connected ? "bg-emerald-400 animate-pulse" : "bg-red-400"
        }`}
      />
      {label || (connected ? "Connected" : "Disconnected")}
    </span>
  );
};

