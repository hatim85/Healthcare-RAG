import React, { useState, useRef, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Stethoscope,
  Send,
  Trash2,
  Sparkles,
  Bot,
  User,
  ShieldCheck,
  Cpu,
  Clock,
  LogIn,
  FileText,
} from "lucide-react";
import { useAppDispatch, useAppSelector } from "@/app/hooks";
import {
  addMessage,
  setLoading,
  clearMessages,
} from "@/features/clinical/clinicalSlice";
import { setCredentials } from "@/features/auth/authSlice";
import { useAskQuestionMutation, useLoginMutation } from "@/app/api";
import { SourceCard } from "@/components/SourceCard";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorAlert } from "@/components/ErrorAlert";
import type { AskResponse } from "@/types";

interface MessageItem {
  role: "user" | "assistant";
  content: string;
  responseMeta?: AskResponse;
}

export const ClinicalChatPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const initialPromptParam = searchParams.get("prompt");

  const dispatch = useAppDispatch();
  const auth = useAppSelector((state) => state.auth);

  const [askQuestion, { isLoading: isAsking }] = useAskQuestionMutation();
  const [login, { isLoading: isLoggingIn }] = useLoginMutation();

  const [inputQuery, setInputQuery] = useState("");
  const [chatHistory, setChatHistory] = useState<MessageItem[]>([]);
  const [loginUsername, setLoginUsername] = useState("clinician");
  const [loginPassword, setLoginPassword] = useState("change_clinician_password_123");
  const [loginError, setLoginError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (initialPromptParam) {
      setInputQuery(
        initialPromptParam === "clinical_consultation"
          ? "What is the first-line pharmacological therapy for newly diagnosed adult type 2 diabetes?"
          : initialPromptParam
      );
    }
  }, [initialPromptParam]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, isAsking]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError(null);
    try {
      const res = await login({
        username: loginUsername,
        password: loginPassword,
      }).unwrap();
      dispatch(setCredentials(res));
    } catch (err: any) {
      setLoginError(
        err?.data?.detail || "Invalid login credentials. Please check password."
      );
    }
  };

  const handleSend = async () => {
    const q = inputQuery.trim();
    if (!q || isAsking) return;

    // Add user message to view
    const userMsg: MessageItem = { role: "user", content: q };
    setChatHistory((prev) => [...prev, userMsg]);
    dispatch(addMessage({ role: "user", content: q }));
    setInputQuery("");
    dispatch(setLoading(true));

    try {
      const result = await askQuestion({
        question: q,
        verbose: false,
      }).unwrap();

      const assistantMsg: MessageItem = {
        role: "assistant",
        content: result.answer,
        responseMeta: result,
      };

      setChatHistory((prev) => [...prev, assistantMsg]);
      dispatch(addMessage({ role: "assistant", content: result.answer }));
    } catch (err: any) {
      const errMsg: MessageItem = {
        role: "assistant",
        content:
          err?.data?.detail ||
          err?.message ||
          "An error occurred while evaluating the clinical question.",
      };
      setChatHistory((prev) => [...prev, errMsg]);
    } finally {
      dispatch(setLoading(false));
    }
  };

  const handleClear = () => {
    setChatHistory([]);
    dispatch(clearMessages());
  };

  const samplePrompts = [
    "What is the first-line therapy for type 2 diabetes according to guidelines?",
    "What are the diagnostic thresholds and monitoring parameters for hypertension?",
    "What are clinical contraindications for metformin in renal impairment?",
  ];

  // If user is not authenticated, show a login form
  if (!auth.token) {
    return (
      <div className="max-w-md mx-auto my-12 p-8 bg-slate-900 border border-slate-800 rounded-3xl shadow-2xl">
        <div className="text-center mb-6">
          <div className="w-12 h-12 rounded-2xl bg-sky-500/15 border border-sky-500/30 flex items-center justify-center text-sky-400 mx-auto mb-3">
            <Stethoscope className="w-6 h-6" />
          </div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Clinician Access
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Sign in with JWT credentials to query evidence-grounded guidelines
          </p>
        </div>

        {loginError && (
          <div className="mb-4">
            <ErrorAlert
              message={loginError}
              onDismiss={() => setLoginError(null)}
            />
          </div>
        )}

        <form onSubmit={handleLogin} className="space-y-4 text-xs">
          <div>
            <label className="block text-slate-300 font-semibold mb-1 uppercase tracking-wider">
              Username
            </label>
            <input
              type="text"
              value={loginUsername}
              onChange={(e) => setLoginUsername(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 focus:border-sky-500 rounded-xl px-3.5 py-2.5 text-white outline-none transition-colors"
              required
            />
          </div>

          <div>
            <label className="block text-slate-300 font-semibold mb-1 uppercase tracking-wider">
              Password
            </label>
            <input
              type="password"
              value={loginPassword}
              onChange={(e) => setLoginPassword(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 focus:border-sky-500 rounded-xl px-3.5 py-2.5 text-white outline-none transition-colors"
              required
            />
          </div>

          <div className="pt-2">
            <button
              type="submit"
              disabled={isLoggingIn}
              className="w-full py-2.5 bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-semibold rounded-xl transition-all shadow-lg shadow-sky-900/30 flex items-center justify-center gap-2 cursor-pointer"
            >
              {isLoggingIn ? (
                <>Signing In...</>
              ) : (
                <>
                  <LogIn className="w-4 h-4" /> Sign In
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-[calc(100vh-6.5rem)]">
      {/* Chat Header */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-800 mb-4 shrink-0">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight flex items-center gap-2">
            <Stethoscope className="w-6 h-6 text-sky-400" />
            Evidence-Grounded Clinical Consultation
          </h2>
          <p className="text-slate-400 text-xs mt-0.5">
            Protected by PII Anonymization, NeMo Safety Rails, Hybrid RRF & Claim Faithfulness
          </p>
        </div>

        {chatHistory.length > 0 && (
          <button
            onClick={handleClear}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-red-400 rounded-lg text-xs transition-colors border border-slate-800 cursor-pointer"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear
          </button>
        )}
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto space-y-5 pr-2">
        {chatHistory.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-lg mx-auto py-12">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center shadow-xl shadow-sky-500/20 mb-4">
              <Bot className="w-7 h-7 text-white" />
            </div>
            <h3 className="text-lg font-bold text-white">
              Clinical Guidelines Assistant
            </h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Ask medical questions backed by indexed guideline PDFs. Every statement is checked for cited evidence with strict claim verification.
            </p>

            {/* Quick Sample Prompts */}
            <div className="w-full mt-6 space-y-2 text-left">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block text-center mb-1">
                Suggested Questions
              </span>
              {samplePrompts.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => setInputQuery(prompt)}
                  className="w-full text-left p-3 rounded-xl bg-slate-900/80 hover:bg-slate-800 border border-slate-800/80 hover:border-sky-500/40 text-xs text-slate-300 transition-all flex items-start gap-2.5 cursor-pointer"
                >
                  <Sparkles className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
                  <span>{prompt}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          chatHistory.map((msg, index) => {
            const isUser = msg.role === "user";
            const meta = msg.responseMeta;
            const sources = meta?.sources || [];

            return (
              <div
                key={index}
                className={`flex gap-3 ${
                  isUser ? "justify-end" : "justify-start"
                }`}
              >
                {!isUser && (
                  <div className="w-8 h-8 rounded-xl bg-sky-600/20 border border-sky-500/40 flex items-center justify-center text-sky-400 shrink-0 mt-1">
                    <Bot className="w-4 h-4" />
                  </div>
                )}

                <div
                  className={`max-w-[85%] rounded-2xl p-4 text-sm leading-relaxed ${
                    isUser
                      ? "bg-sky-600 text-white rounded-br-xs shadow-md shadow-sky-950/40"
                      : "bg-slate-900 border border-slate-800 text-slate-200 rounded-bl-xs shadow-lg"
                  }`}
                >
                  <div className="whitespace-pre-wrap">{msg.content}</div>

                  {/* Supporting Sources / Citations */}
                  {!isUser && sources.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-slate-800/90 space-y-2">
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
                        <FileText className="w-3.5 h-3.5 text-sky-400" />
                        Supporting Evidence ({sources.length} sources):
                      </div>
                      <div className="space-y-2">
                        {sources.map((src, sIdx) => (
                          <SourceCard
                            key={sIdx}
                            sourceId={src.source_id || `S${sIdx + 1}`}
                            filename={src.filename}
                            page={src.page}
                            text={src.text}
                          />
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Observability & Verification Metadata */}
                  {!isUser && meta && (
                    <div className="mt-3 pt-2 border-t border-slate-800/80 flex flex-wrap items-center gap-3 text-[11px] text-slate-400">
                      {meta.model && (
                        <span className="flex items-center gap-1">
                          <Cpu className="w-3 h-3 text-slate-500" />
                          Model: <code className="text-slate-300">{meta.model}</code>
                        </span>
                      )}
                      {meta.citation_check && (
                        <span className="flex items-center gap-1 text-emerald-400">
                          <ShieldCheck className="w-3 h-3" />
                          Citations Verified
                        </span>
                      )}
                      {meta.observability && (
                        <span className="flex items-center gap-1 text-slate-400">
                          <Clock className="w-3 h-3 text-slate-500" />
                          {(meta.observability.total_latency_ms as number)?.toFixed(0) || 0} ms
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {isUser && (
                  <div className="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 shrink-0 mt-1">
                    <User className="w-4 h-4" />
                  </div>
                )}
              </div>
            );
          })
        )}

        {isAsking && (
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-sky-600/20 border border-sky-500/40 flex items-center justify-center text-sky-400">
              <Bot className="w-4 h-4" />
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 text-xs text-slate-400 flex items-center gap-2">
              <LoadingSpinner message="Consulting guidelines (Retrieval, Reranking & Claim Verification)..." size={16} />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input Bar */}
      <div className="pt-3 border-t border-slate-800">
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 focus-within:border-sky-500 rounded-2xl p-2 transition-all shadow-xl">
          <input
            type="text"
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask a clinical question grounded in medical guidelines..."
            className="flex-1 bg-transparent px-3 py-1.5 text-sm text-white placeholder-slate-500 outline-none"
            disabled={isAsking}
          />

          <button
            onClick={handleSend}
            disabled={isAsking || !inputQuery.trim()}
            className="p-2.5 bg-sky-600 hover:bg-sky-500 disabled:bg-slate-800 disabled:text-slate-600 text-white rounded-xl transition-all shadow-md shadow-sky-950 cursor-pointer"
            aria-label="Send clinical query"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
