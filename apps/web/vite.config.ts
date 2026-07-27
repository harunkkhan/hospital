import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev proxy: the FastAPI backend (apps/api) serves at :8000 with routes at the
// root (/runs, /scenarios). The web client always talks to /api/* so the same
// build works behind any reverse proxy; in dev, Vite strips the prefix.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
