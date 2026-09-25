import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

const API = process.env.TRUSTRAG_URL || "http://127.0.0.1:8024";

export default defineConfig({
  plugins: [vue()],
  // build 出來的東西由 FastAPI 掛在 /ui/，資源路徑要從那裡算
  base: "/ui/",
  build: {
    outDir: "../day24_service/static",
    emptyOutDir: true,
  },
  server: {
    // 開發時 Vite 在 5173、服務在 8024，不同源。用 proxy 轉過去，
    // 瀏覽器看到的永遠是同源，服務就不必為了開發開 CORS
    proxy: {
      "/ask": API,
      "/search": API,
      "/sources": API,
      "/healthz": API,
    },
  },
});
