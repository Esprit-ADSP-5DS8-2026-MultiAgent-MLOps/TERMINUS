import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// En développement, /api est relayé vers l'API de Terminus (python -m orchestrator api).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": process.env.TERMINUS_API ?? "http://127.0.0.1:8090" },
  },
});
