import React from "react";
import { Link } from "react-router-dom";
import {
  Server,
  Layers,
  Database,
  RefreshCw,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  Stethoscope,
  Wrench,
  Activity,
  HardDrive,
} from "lucide-react";
import { useGetProviderStatusQuery, useGetHealthQuery } from "@/app/api";
import { StatusBadge } from "@/components/StatusBadge";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorAlert } from "@/components/ErrorAlert";

export const DashboardPage: React.FC = () => {
  const {
    data: providers,
    isLoading: loadingProviders,
    error: providersError,
    refetch: refetchProviders,
  } = useGetProviderStatusQuery();

  const {
    data: health,
    isLoading: loadingHealth,
    error: healthError,
    refetch: refetchHealth,
  } = useGetHealthQuery();

  const handleRefreshAll = () => {
    refetchProviders();
    refetchHealth();
  };

  const totalProviders = providers?.length || 0;
  const connectedCount = providers?.filter((p) => p.connected).length || 0;
  const totalTools =
    providers?.reduce((acc, p) => acc + (p.tools_count || 0), 0) || 0;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <Server className="w-8 h-8 text-sky-400" />
            MCP Provider Hub
          </h2>
          <p className="text-slate-400 text-sm mt-1">
            Real-time multi-provider orchestration & health monitoring
          </p>
        </div>

        <button
          onClick={handleRefreshAll}
          className="self-start sm:self-auto inline-flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-semibold transition-all border border-slate-700 cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5 text-sky-400" />
          Refresh Providers
        </button>
      </div>

      {providersError && (
        <ErrorAlert message="Failed to load provider status from backend." />
      )}
      {healthError && (
        <ErrorAlert message="Failed to load system health checks." />
      )}

      {/* Metric Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 bg-slate-900 border border-slate-800 rounded-2xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase font-semibold tracking-wider">
              Connected Providers
            </span>
            <Server className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-bold text-white flex items-baseline gap-2">
            {connectedCount}
            <span className="text-sm font-normal text-slate-400">
              / {totalProviders} configured
            </span>
          </div>
          <div className="text-xs text-slate-400 mt-2 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            Live MCP sessions
          </div>
        </div>

        <div className="p-5 bg-slate-900 border border-slate-800 rounded-2xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase font-semibold tracking-wider">
              Discovered Tools
            </span>
            <Wrench className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-3xl font-bold text-white">{totalTools}</div>
          <div className="text-xs text-slate-400 mt-2">
            Callable across all services
          </div>
        </div>

        <div className="p-5 bg-slate-900 border border-slate-800 rounded-2xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase font-semibold tracking-wider">
              Vector DB (Qdrant)
            </span>
            <Database className="w-4 h-4 text-rose-400" />
          </div>
          <div className="text-2xl font-bold text-white capitalize flex items-center gap-2">
            {loadingHealth ? (
              "Checking..."
            ) : health?.qdrant === "ok" ? (
              <>
                <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                Connected
              </>
            ) : healthError ? (
              <>
                <AlertTriangle className="w-6 h-6 text-amber-400" />
                Unavailable
              </>
            ) : (
              <>
                <AlertTriangle className="w-6 h-6 text-amber-400" />
                {health?.qdrant || "Degraded"}
              </>
            )}
          </div>
          <div className="text-xs text-slate-400 mt-2 truncate">
            Embeddings & clinical chunks
          </div>
        </div>

        <div className="p-5 bg-slate-900 border border-slate-800 rounded-2xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase font-semibold tracking-wider">
              System Health
            </span>
            <Activity className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-bold text-white capitalize flex items-center gap-2">
            {loadingHealth ? (
              "Checking..."
            ) : health?.status === "healthy" ? (
              <>
                <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                Healthy
              </>
            ) : healthError ? (
              <>
                <AlertTriangle className="w-6 h-6 text-rose-400" />
                Error
              </>
            ) : (
              <>
                <AlertTriangle className="w-6 h-6 text-amber-400" />
                {health?.status || "Degraded"}
              </>
            )}
          </div>
          <div className="text-xs text-slate-400 mt-2">
            Active indexing jobs: {health?.active_jobs ?? 0}
          </div>
        </div>
      </div>

      {/* Provider Status Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Layers className="w-5 h-5 text-sky-400" />
            Configured MCP Providers
          </h3>
          <Link
            to="/tools"
            className="text-xs text-sky-400 hover:text-sky-300 font-medium flex items-center gap-1"
          >
            Explore all tools <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        {loadingProviders ? (
          <LoadingSpinner message="Querying provider status..." />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {providers?.map((provider) => (
              <div
                key={provider.name}
                className="p-5 bg-slate-900/90 border border-slate-800 hover:border-slate-700 rounded-2xl transition-all shadow-md flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-base font-bold text-white font-mono uppercase tracking-wide">
                      {provider.name}
                    </span>
                    <StatusBadge
                      connected={provider.connected}
                      label={
                        provider.connected
                          ? "ONLINE"
                          : provider.enabled
                          ? "FAILED"
                          : "DISABLED"
                      }
                      size="sm"
                    />
                  </div>

                  <div className="space-y-2 text-xs text-slate-400 mb-4">
                    <div className="flex justify-between py-1 border-b border-slate-800/80">
                      <span>Transport:</span>
                      <span className="font-mono text-slate-300">
                        {provider.transport}
                      </span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/80">
                      <span>Enabled in .env:</span>
                      <span
                        className={
                          provider.enabled
                            ? "text-emerald-400 font-medium"
                            : "text-slate-500 font-medium"
                        }
                      >
                        {provider.enabled ? "True" : "False"}
                      </span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/80">
                      <span>Tools Available:</span>
                      <span className="font-mono text-slate-200 font-semibold">
                        {provider.tools_count}
                      </span>
                    </div>
                    {provider.error && (
                      <div className="text-red-400 text-[11px] pt-1 leading-snug">
                        {provider.error}
                      </div>
                    )}
                  </div>
                </div>

                <Link
                  to={`/tools?provider=${provider.name}`}
                  className="mt-2 w-full py-2 px-3 bg-slate-800/90 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-xl text-center transition-colors flex items-center justify-center gap-1.5"
                >
                  <Wrench className="w-3.5 h-3.5 text-sky-400" />
                  View {provider.name} Tools
                </Link>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Backend Infrastructure Overview */}
      <div className="p-6 bg-slate-900 border border-slate-800 rounded-2xl space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <HardDrive className="w-5 h-5 text-indigo-400" />
          Healthcare RAG Engine Telemetry
        </h3>

        {loadingHealth ? (
          <LoadingSpinner message="Checking engine telemetry..." />
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
            <div className="p-3.5 bg-slate-950/60 border border-slate-800 rounded-xl">
              <span className="text-slate-400 block mb-1">Document Registry:</span>
              <span className="font-semibold text-slate-200">
                {health?.sqlite_registry || "OK"} ({health?.database || "sqlite"})
              </span>
            </div>
            <div className="p-3.5 bg-slate-950/60 border border-slate-800 rounded-xl">
              <span className="text-slate-400 block mb-1">Redis Cache Store:</span>
              <span className="font-semibold text-slate-200">
                {health?.redis || "OK"}
              </span>
            </div>
            <div className="p-3.5 bg-slate-950/60 border border-slate-800 rounded-xl">
              <span className="text-slate-400 block mb-1">Active Indexing Workers:</span>
              <span className="font-semibold text-slate-200">
                {health?.active_jobs ?? 0} jobs in progress
              </span>
            </div>
            <div className="p-3.5 bg-slate-950/60 border border-slate-800 rounded-xl">
              <span className="text-slate-400 block mb-1">Last Heartbeat:</span>
              <span className="font-semibold text-slate-200 truncate block">
                {health?.timestamp ? new Date(health.timestamp).toLocaleTimeString() : "N/A"}
              </span>
            </div>
          </div>
        )}

        <div className="pt-2 flex flex-wrap gap-3">
          <Link
            to="/clinical"
            className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-medium rounded-xl text-xs transition-all shadow-md shadow-sky-900/30"
          >
            <Stethoscope className="w-4 h-4" />
            Launch Clinical Consultation
          </Link>
          <Link
            to="/prompts"
            className="inline-flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium rounded-xl text-xs transition-all border border-slate-700"
          >
            Explore MCP Prompts
          </Link>
        </div>
      </div>
    </div>
  );
};
