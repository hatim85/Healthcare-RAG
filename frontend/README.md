# Healthcare Clinical RAG — React + Redux Toolkit Frontend

Modern React SPA dashboard and clinical consultation interface built with **React 19**, **Redux Toolkit (RTK Query)**, **TypeScript**, **Tailwind CSS 4**, and **Lucide Icons**.

Seamlessly integrates with the Healthcare RAG FastAPI server (`http://healthcare.localhost`) and the Multi-Provider Model Context Protocol (MCP) framework.

---

## 🚀 Key Features

1. **MCP Provider Dashboard (`/`)**
   - Live connectivity status for all configured providers: **Langfuse**, **Qdrant**, **Firecrawl**, **Groq**, **LiteLLM**, and **Redis**.
   - Displays transport type (`streamable-http`, `stdio`), active tool counts, and connection health.
   - Real-time engine telemetry (Qdrant vector DB, SQLite document registry, Redis cache, active background indexing jobs).

2. **Interactive MCP Tools Explorer (`/tools`)**
   - Search and filter tools across all connected external MCP servers.
   - Interactive Tool Execution Modal: enter JSON arguments and trigger live tool execution against external services.
   - Inspect formatted responses, copy output, and review error statuses.

3. **MCP Prompts Catalog (`/prompts`)**
   - Browse prompt templates registered across MCP providers (e.g. Langfuse, Clinical Consultation).
   - One-click copy or load directly into the Clinical Chat.

4. **MCP Resources Catalog (`/resources`)**
   - Inspect data feeds and state endpoints (e.g., `guidelines://catalog`, `health://status`).

5. **Clinical Guidelines Consultation Chat (`/clinical`)**
   - JWT-authenticated clinical consultation guarded by:
     - Presidio PII Anonymization
     - NeMo Input & Output Safety Rails
     - Hybrid Vector + Lexical Search (RRF)
     - Cross-Encoder Reranker
     - Dynamic Context Budgeting
     - Citation Grounding ([S1], [S2]) & Claim Verification
   - Displays supporting guideline source cards with page references and expandable excerpts.

---

## 🛠️ Architecture & Tech Stack

- **Framework:** React 19 + Vite 6 + TypeScript
- **State Management & Data Fetching:** Redux Toolkit (`@reduxjs/toolkit`) + RTK Query (`createApi`, `fetchBaseQuery`)
- **Routing:** React Router v7 (`react-router-dom`)
- **Styling:** Tailwind CSS v4 (`@tailwindcss/vite`)
- **Icons:** Lucide React (`lucide-react`)
- **API Proxy:** Vite development proxy forwarding `/api` and `/auth` to FastAPI (`http://healthcare.localhost`)

---

## 📦 Getting Started

### 1. Ensure the Backend is Running

In your backend terminal:
```bash
# From workspace root
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Install & Start Frontend

In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```

Open your browser at **http://localhost:3000**.

### 3. Clinician Login (for Clinical Chat)

Default credentials configured in `.env.example`:
- **Username:** `clinician`
- **Password:** `change_clinician_password_123`

---

## 📂 Project Structure

```
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── src/
│   ├── main.tsx                  # App entrypoint with Redux Provider
│   ├── App.tsx                   # Main router view with Layout
│   ├── store.ts                  # Redux Toolkit store configuration
│   ├── index.css                 # Tailwind CSS 4 directives
│   ├── types/
│   │   └── index.ts              # Full TypeScript API definitions
│   ├── app/
│   │   ├── api.ts                # RTK Query API endpoints & hooks
│   │   └── hooks.ts              # Typed Redux useAppDispatch / useAppSelector
│   ├── features/
│   │   ├── auth/authSlice.ts     # JWT authentication & session slice
│   │   └── clinical/clinicalSlice.ts # Chat conversation state slice
│   ├── components/
│   │   ├── Layout.tsx            # App shell with responsive sidebar & brand header
│   │   ├── StatusBadge.tsx       # Live status indicators (Online/Offline)
│   │   ├── ToolCard.tsx          # MCP tool display card
│   │   ├── SourceCard.tsx        # Clinical evidence citation card
│   │   ├── LoadingSpinner.tsx    # Loading state indicator
│   │   └── ErrorAlert.tsx        # Error boundary/banner
│   └── pages/
│       ├── DashboardPage.tsx     # Provider metrics & engine telemetry
│       ├── ToolsPage.tsx         # MCP tools search & interactive execution
│       ├── PromptsPage.tsx       # Prompt template catalog
│       ├── ResourcesPage.tsx     # Resource URIs
│       └── ClinicalChatPage.tsx  # Clinical consultation Q&A
```

