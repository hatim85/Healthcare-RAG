import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://healthcare.localhost",
        changeOrigin: true,
      },
      "/auth": {
        target: "http://healthcare.localhost",
        changeOrigin: true,
      },
      "/health": {
        target: "http://healthcare.localhost",
        changeOrigin: true,
      },
      "/ask": {
        target: "http://healthcare.localhost",
        changeOrigin: true,
      },
      "/mcp": {
        target: "http://healthcare.localhost",
        changeOrigin: true,
      },
    },
  },
});
