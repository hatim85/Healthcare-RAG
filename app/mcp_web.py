"""MCP-Compatible Web Chat Interface for Healthcare RAG.

Serves an interactive web application allowing users without Claude Desktop to:
1. Inspect discovered MCP tools, resources, and prompts.
2. Ask clinical questions that run through the MCP protocol wrapping AnswerPipeline.
3. View verified citations ([S1], [S2]), source passages, and observability latency.
4. Directly test MCP SSE transport.
"""

from contextlib import asynccontextmanager
import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.mcp_server import mcp_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_web_chat")

app = FastAPI(
    title="Healthcare RAG — MCP Web Chat Interface",
    description="Interactive web chat interface powered by Model Context Protocol (MCP)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ToolCallRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the MCP tool to execute")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments dictionary")


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Healthcare RAG — MCP Web Chat</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
  <style>
    :root {
      --bg-dark: #0f172a;
      --panel-dark: #1e293b;
      --border-dark: #334155;
      --accent: #0ea5e9;
      --accent-hover: #0284c7;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #10b981;
      --card-bg: #1e293b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg-dark); color: var(--text); height: 100vh; display: flex; overflow: hidden; }
    
    /* Sidebar */
    .sidebar {
      width: 340px;
      background: var(--panel-dark);
      border-right: 1px solid var(--border-dark);
      display: flex;
      flex-direction: column;
      padding: 20px;
      overflow-y: auto;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border-dark);
    }
    .brand i { font-size: 1.8rem; color: var(--accent); }
    .brand h2 { font-size: 1.15rem; font-weight: 700; color: #fff; }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.75rem;
      background: rgba(16, 185, 129, 0.15);
      color: var(--success);
      margin-bottom: 20px;
    }
    .status-badge .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); }
    
    .section-title { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.05em; margin-bottom: 12px; font-weight: 600; }
    
    .tool-card {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-dark);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 10px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .tool-card:hover {
      border-color: var(--accent);
      background: rgba(14, 165, 233, 0.08);
    }
    .tool-name { font-weight: 600; font-size: 0.85rem; color: #38bdf8; display: flex; align-items: center; gap: 6px; }
    .tool-desc { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; line-height: 1.3; }

    /* Main Chat Area */
    .chat-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: var(--bg-dark);
    }
    .chat-header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--border-dark);
      background: rgba(30, 41, 59, 0.4);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .chat-header h1 { font-size: 1.1rem; font-weight: 600; }
    .protocol-tag {
      background: rgba(14, 165, 233, 0.15);
      color: var(--accent);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 500;
    }
    
    .chat-messages {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    
    .message {
      max-width: 85%;
      padding: 14px 18px;
      border-radius: 12px;
      font-size: 0.95rem;
      line-height: 1.5;
    }
    .user-msg {
      align-self: flex-end;
      background: var(--accent);
      color: #fff;
      border-bottom-right-radius: 2px;
    }
    .assistant-msg {
      align-self: flex-start;
      background: var(--panel-dark);
      border: 1px solid var(--border-dark);
      color: var(--text);
      border-bottom-left-radius: 2px;
      white-space: pre-wrap;
    }
    .assistant-msg strong { color: #38bdf8; }
    .assistant-msg code { background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px; font-family: monospace; }
    
    /* Quick prompts */
    .quick-prompts {
      display: flex;
      gap: 8px;
      padding: 0 24px 12px 24px;
      overflow-x: auto;
    }
    .prompt-chip {
      background: var(--panel-dark);
      border: 1px solid var(--border-dark);
      color: var(--text-muted);
      padding: 6px 12px;
      border-radius: 9999px;
      font-size: 0.8rem;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s;
    }
    .prompt-chip:hover {
      border-color: var(--accent);
      color: #fff;
    }

    /* Input area */
    .chat-input-area {
      padding: 16px 24px;
      border-top: 1px solid var(--border-dark);
      background: var(--panel-dark);
      display: flex;
      gap: 12px;
    }
    .chat-input {
      flex: 1;
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--border-dark);
      border-radius: 8px;
      padding: 12px 16px;
      color: #fff;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s;
    }
    .chat-input:focus { border-color: var(--accent); }
    .send-btn {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 0 20px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: background 0.2s;
    }
    .send-btn:hover { background: var(--accent-hover); }
    .send-btn:disabled { opacity: 0.6; cursor: not-allowed; }
  </style>
</head>
<body>

  <!-- Sidebar -->
  <div class="sidebar">
    <div class="brand">
      <i class="bi bi-heart-pulse-fill"></i>
      <div>
        <h2>Clinical RAG MCP</h2>
        <div style="font-size:0.75rem; color:var(--text-muted);">Model Context Protocol</div>
      </div>
    </div>
    
    <div class="status-badge">
      <div class="dot"></div>
      <span>MCP Server Ready (stdio / SSE)</span>
    </div>

    <div class="section-title">Discovered MCP Tools</div>
    <div id="tools-list">
      <div style="color:var(--text-muted); font-size:0.8rem;">Loading discovered tools...</div>
    </div>

    <div class="section-title" style="margin-top:20px;">MCP Resources</div>
    <div class="tool-card" onclick="runTool('list_guideline_documents', {})">
      <div class="tool-name"><i class="bi bi-file-earmark-medical"></i> guidelines://catalog</div>
      <div class="tool-desc">Catalog of registered clinical guideline documents</div>
    </div>
    <div class="tool-card" onclick="runTool('check_system_health', {})">
      <div class="tool-name"><i class="bi bi-activity"></i> health://status</div>
      <div class="tool-desc">Real-time health check for Qdrant and SQLite</div>
    </div>
  </div>

  <!-- Main Chat -->
  <div class="chat-container">
    <div class="chat-header">
      <div>
        <h1>Evidence-Grounded Clinical Consultation</h1>
        <div style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">
          Guarded by PII Anonymization, NeMo Safety, Hybrid Reranking & Claim Verification
        </div>
      </div>
      <div class="protocol-tag"><i class="bi bi-cpu"></i> MCP FastMCP 2.0</div>
    </div>

    <div class="chat-messages" id="chat-messages">
      <div class="message assistant-msg">
        👋 Welcome to the <strong>Healthcare Clinical RAG Assistant</strong> via Model Context Protocol!

You can ask evidence-grounded questions regarding medical guidelines, inspect retrieved passages, or verify system health. All clinical answers include explicit citations (such as <code>[S1]</code>) and pass through rigorous claim verification.
      </div>
    </div>

    <!-- Quick Prompts -->
    <div class="quick-prompts">
      <div class="prompt-chip" onclick="setQuery('What is the initial therapy for type 2 diabetes?')">
        💊 Initial diabetes therapy?
      </div>
      <div class="prompt-chip" onclick="runTool('check_system_health', {})">
        🩺 System Health Check
      </div>
      <div class="prompt-chip" onclick="runTool('list_guideline_documents', {limit: 10})">
        📑 List Guideline PDFs
      </div>
      <div class="prompt-chip" onclick="setQuery('What are symptoms and complications of hypertension?')">
        ❤️ Hypertension symptoms?
      </div>
    </div>

    <!-- Input -->
    <div class="chat-input-area">
      <input type="text" id="chat-input" class="chat-input" placeholder="Type a clinical question grounded in guidelines..." onkeydown="if(event.key==='Enter') sendQuestion()">
      <button class="send-btn" id="send-btn" onclick="sendQuestion()">
        <span>Consult</span>
        <i class="bi bi-send-fill"></i>
      </button>
    </div>
  </div>

  <script>
    const messagesContainer = document.getElementById('chat-messages');
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const toolsList = document.getElementById('tools-list');

    // Fetch and display discovered tools from MCP Server
    async function loadTools() {
      try {
        const res = await fetch('/api/mcp/tools');
        const tools = await res.json();
        toolsList.innerHTML = '';
        tools.forEach(t => {
          const div = document.createElement('div');
          div.className = 'tool-card';
          div.onclick = () => {
            if (t.name === 'check_system_health') runTool('check_system_health', {});
            else if (t.name === 'list_guideline_documents') runTool('list_guideline_documents', {limit: 10});
            else {
              input.placeholder = `Run MCP tool '${t.name}'...`;
              input.focus();
            }
          };
          div.innerHTML = `
            <div class="tool-name"><i class="bi bi-tools"></i> ${t.name}</div>
            <div class="tool-desc">${(t.description || '').slice(0, 80)}...</div>
          `;
          toolsList.appendChild(div);
        });
      } catch (e) {
        toolsList.innerHTML = '<div style="color:red; font-size:0.8rem;">Failed to load tools</div>';
      }
    }

    function appendMessage(text, isUser = false) {
      const msg = document.createElement('div');
      msg.className = `message ${isUser ? 'user-msg' : 'assistant-msg'}`;
      msg.innerText = text;
      messagesContainer.appendChild(msg);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
      return msg;
    }

    function setQuery(text) {
      input.value = text;
      input.focus();
    }

    async function sendQuestion() {
      const q = input.value.trim();
      if (!q) return;

      appendMessage(q, true);
      input.value = '';
      sendBtn.disabled = true;

      const loadingMsg = appendMessage("Thinking... (Querying AnswerPipeline over MCP protocol)");

      try {
        const res = await fetch('/api/mcp/call', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            tool_name: 'ask_clinical_question',
            arguments: { question: q }
          })
        });
        const data = await res.json();
        loadingMsg.innerText = data.result || data.detail || "No result returned";
      } catch (err) {
        loadingMsg.innerText = `Error contacting MCP Server: ${err.message}`;
      } finally {
        sendBtn.disabled = false;
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }
    }

    async function runTool(toolName, args) {
      appendMessage(`[MCP Tool Request] ${toolName}(${JSON.stringify(args)})`, true);
      const loadingMsg = appendMessage(`Executing MCP Tool '${toolName}'...`);
      try {
        const res = await fetch('/api/mcp/call', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tool_name: toolName, arguments: args })
        });
        const data = await res.json();
        loadingMsg.innerText = data.result || JSON.stringify(data, null, 2);
      } catch (err) {
        loadingMsg.innerText = `Tool call error: ${err.message}`;
      }
    }

    loadTools();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, summary="MCP Web Chat Interface UI")
def serve_web_ui():
    """Serve the interactive MCP web chat interface."""
    return HTML_PAGE


@app.get("/api/mcp/tools", summary="List discovered MCP Tools")
async def list_mcp_tools():
    """Return all tools registered in the FastMCP server."""
    tools = await mcp_server.list_tools()
    return [
        {
            "name": t.name,
            "description": t.description,
            "inputSchema": getattr(t, "inputSchema", {}),
        }
        for t in tools
    ]


@app.post("/api/mcp/call", summary="Execute an MCP tool")
async def call_mcp_tool(req: ToolCallRequest):
    """Call a tool via the MCP server protocol and return the result."""
    try:
        call_result = await mcp_server.call_tool(req.tool_name, req.arguments)
        text_output = ""
        for item in call_result.content:
            if hasattr(item, "text"):
                text_output += item.text
        return {
            "tool_name": req.tool_name,
            "is_error": call_result.is_error,
            "result": text_output,
        }
    except Exception as exc:
        logger.error(f"Failed to call MCP tool '{req.tool_name}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"MCP Tool execution failed: {exc}",
        )


def main():
    import uvicorn
    print("\n" + "=" * 70)
    print("STARTING HEALTHCARE RAG MCP WEB CHAT INTERFACE")
    print("Open your browser at: http://127.0.0.1:8002")
    print("=" * 70 + "\n")
    uvicorn.run("app.mcp_web:app", host="127.0.0.1", port=8002, reload=False)


if __name__ == "__main__":
    main()

