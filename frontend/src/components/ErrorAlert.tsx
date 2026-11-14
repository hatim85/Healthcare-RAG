import React from "react";
import { AlertCircle, X } from "lucide-react";

interface ErrorAlertProps {
  message: string;
  onDismiss?: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({ message, onDismiss }) => {
  return (
    <div className="flex items-start justify-between gap-3 p-4 bg-red-950/40 border border-red-500/40 rounded-xl text-red-200 text-sm">
      <div className="flex items-start gap-2.5">
        <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
        <span className="leading-relaxed">{message}</span>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="text-red-400 hover:text-red-200 transition-colors p-1"
          aria-label="Dismiss error"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};

