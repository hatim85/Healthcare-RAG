import React from "react";
import { Loader2 } from "lucide-react";

interface LoadingSpinnerProps {
  message?: string;
  size?: number;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = "Loading...",
  size = 24,
}) => {
  return (
    <div className="flex flex-col items-center justify-center py-12 gap-3 text-slate-400">
      <Loader2 size={size} className="animate-spin text-sky-400" />
      {message && <p className="text-sm font-medium">{message}</p>}
    </div>
  );
};

