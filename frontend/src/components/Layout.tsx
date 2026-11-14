import React, { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  Wrench,
  MessageSquare,
  FolderOpen,
  Stethoscope,
  HeartPulse,
  LogOut,
  LogIn,
  Menu,
  X,
  Activity,
} from "lucide-react";
import { useAppDispatch, useAppSelector } from "@/app/hooks";
import { logout } from "@/features/auth/authSlice";
import { useGetHealthQuery } from "@/app/api";
import { StatusBadge } from "@/components/StatusBadge";

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  const auth = useAppSelector((state) => state.auth);
  const { data: health } = useGetHealthQuery(undefined, {
    pollingInterval: 30000,
  });
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleLogout = () => {
    dispatch(logout());
    navigate("/clinical");
  };

  const navLinks = [
    {
      to: "/",
      label: "Dashboard",
      icon: LayoutDashboard,
      description: "Overview & Providers",
    },
    {
      to: "/tools",
      label: "MCP Tools",
      icon: Wrench,
      description: "Interactive Explorer",
    },
    {
      to: "/prompts",
      label: "MCP Prompts",
      icon: MessageSquare,
      description: "Prompt Catalog",
    },
    {
      to: "/resources",
      label: "MCP Resources",
      icon: FolderOpen,
      description: "Data Endpoints",
    },
    {
      to: "/clinical",
      label: "Clinical Chat",
      icon: Stethoscope,
      description: "Evidence RAG Q&A",
    },
  ];

  const isHealthy = health?.status === "healthy";

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div
          onClick={() => setMobileOpen(false)}
          className="fixed inset-0 bg-black/60 z-40 lg:hidden backdrop-blur-xs"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed lg:static top-0 bottom-0 left-0 z-50 w-72 bg-slate-900 border-r border-slate-800 flex flex-col justify-between transition-transform duration-200 ease-in-out ${
          mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        }`}
      >
        {/* Brand Header */}
        <div>
          <div className="flex items-center justify-between p-5 border-b border-slate-800">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
                <HeartPulse className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-base tracking-tight text-white flex items-center gap-1.5">
                  Clinical RAG
                  <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-400 border border-sky-500/30">
                    MCP
                  </span>
                </h1>
                <p className="text-xs text-slate-400">Healthcare Assistant</p>
              </div>
            </div>

            <button
              onClick={() => setMobileOpen(false)}
              className="lg:hidden text-slate-400 hover:text-white p-1"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* System Health Widget */}
          <div className="px-5 py-3 bg-slate-900/60 border-b border-slate-800/80 flex items-center justify-between text-xs">
            <span className="flex items-center gap-1.5 text-slate-400">
              <Activity className="w-3.5 h-3.5 text-slate-500" />
              Engine
            </span>
            <StatusBadge
              connected={isHealthy}
              label={health ? health.status.toUpperCase() : "CHECKING..."}
              size="sm"
            />
          </div>

          {/* Navigation Links */}
          <nav className="p-3 space-y-1">
            {navLinks.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === "/"}
                  onClick={() => setMobileOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                      isActive
                        ? "bg-sky-600 text-white shadow-lg shadow-sky-600/30"
                        : "text-slate-300 hover:text-white hover:bg-slate-800/70"
                    }`
                  }
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <div className="flex-1 text-left">
                    <div className="leading-tight">{item.label}</div>
                    <div className="text-[11px] opacity-70 font-normal leading-tight">
                      {item.description}
                    </div>
                  </div>
                </NavLink>
              );
            })}
          </nav>
        </div>

        {/* User / Auth Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-900/40">
          {auth.token ? (
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="font-medium text-xs text-slate-200 truncate">
                  {auth.username}
                </div>
                <div className="text-[11px] text-slate-400 uppercase tracking-wider flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  {auth.role || "user"} • {auth.tenant_id || "default"}
                </div>
              </div>
              <button
                onClick={handleLogout}
                title="Log out"
                className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors cursor-pointer"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <NavLink
              to="/clinical"
              className="flex items-center justify-center gap-2 w-full py-2 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-medium transition-colors"
            >
              <LogIn className="w-3.5 h-3.5 text-sky-400" />
              Sign In (Clinician)
            </NavLink>
          )}
        </div>
      </aside>

      {/* Main View Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Mobile Header Bar */}
        <header className="lg:hidden flex items-center justify-between p-4 bg-slate-900 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <HeartPulse className="w-6 h-6 text-sky-400" />
            <span className="font-bold text-sm">Clinical RAG MCP</span>
          </div>
          <button
            onClick={() => setMobileOpen(true)}
            className="p-2 text-slate-300 hover:text-white rounded-lg bg-slate-800"
          >
            <Menu className="w-5 h-5" />
          </button>
        </header>

        {/* Scrollable View Content */}
        <main className="flex-1 overflow-y-auto bg-slate-950 p-4 sm:p-6 lg:p-8">
          <div className="max-w-7xl mx-auto">{children}</div>
        </main>
      </div>
    </div>
  );
};

