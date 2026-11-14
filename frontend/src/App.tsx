import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { DashboardPage } from "@/pages/DashboardPage";
import { ToolsPage } from "@/pages/ToolsPage";
import { PromptsPage } from "@/pages/PromptsPage";
import { ResourcesPage } from "@/pages/ResourcesPage";
import { ClinicalChatPage } from "@/pages/ClinicalChatPage";

const App: React.FC = () => {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/tools" element={<ToolsPage />} />
        <Route path="/prompts" element={<PromptsPage />} />
        <Route path="/resources" element={<ResourcesPage />} />
        <Route path="/clinical" element={<ClinicalChatPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
};

export default App;

