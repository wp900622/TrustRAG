import { onMounted, reactive } from "vue";

// 頂部那顆燈：服務活著嗎、索引裡有幾塊。/healthz 永遠回 200（Day 24），
// 所以「連不上」與「還在建索引」要分開顯示
export function useHealth() {
  const health = reactive({ status: "checking", chunks: 0, version: "" });

  onMounted(async () => {
    try {
      const r = await fetch("/healthz");
      const b = await r.json();
      Object.assign(health, { status: b.ready ? "ok" : "warming",
                              chunks: b.chunks, version: b.version });
    } catch {
      health.status = "down";
    }
  });

  return health;
}
