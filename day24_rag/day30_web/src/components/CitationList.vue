<script setup>
import { onBeforeUnmount, onMounted, reactive } from "vue";

defineProps({
  items: { type: Array, default: () => [] },
  mode: { type: String, default: "pipeline" },
});

const open = reactive({});

// title 本身就是「第 24 條（病假）」，原文第一行又是標題，內文拿掉第一行
const body = (text) => text.trim().split("\n").slice(1).join("\n").trim() || text;
// 「第 24 條（病假）」拆成徽章的號碼與標題
const heading = (c) => /（(.+)）/.exec(c.title)?.[1] ?? c.title;
const key = (c) => `${c.source}:${c.article_no}`;
// 有沒有被截成三行要量畫面，不能數字數：同一條在手機上會斷、在桌機上不會
const clamped = reactive({});
const els = new Map();
function track(el, c) {
  if (!el) return;
  els.set(key(c), el);
  if (!open[key(c)]) clamped[key(c)] = el.scrollHeight > el.clientHeight + 1;
}
function remeasure() {
  for (const [k, el] of els) if (!open[k]) clamped[k] = el.scrollHeight > el.clientHeight + 1;
}
onMounted(() => window.addEventListener("resize", remeasure));
onBeforeUnmount(() => window.removeEventListener("resize", remeasure));
</script>

<template>
  <section class="cites">
    <header>
      <span class="label">服務查到的條文</span>
      <span class="count">{{ items.length }}</span>
    </header>
    <p class="note">
      不一定每一條都跟答案有關。{{ mode === "agent" ? "這是 agent 查過的每一條。" : "這是檢索撈到的前幾條。" }}
    </p>

    <TransitionGroup name="rise" tag="div" class="list">
      <article v-for="c in items" :key="key(c)" class="card cite" :class="{ open: open[key(c)] }">
        <div class="top">
          <span class="no">第 {{ c.article_no }} 條</span>
          <strong>{{ heading(c) }}</strong>
          <span class="src">{{ c.source }}</span>
        </div>
        <!-- agent 路徑的 similarity 是 0（它不是一次檢索的排名），不畫條 -->
        <div v-if="c.similarity > 0" class="sim" :title="`相似度 ${c.similarity.toFixed(2)}`">
          <span :style="{ width: `${Math.min(c.similarity, 1) * 100}%` }" />
          <small>{{ c.similarity.toFixed(2) }}</small>
        </div>
        <p :ref="(el) => track(el, c)" class="text">{{ body(c.text) }}</p>
        <button v-if="clamped[key(c)] || open[key(c)]" type="button" class="more" @click="open[key(c)] = !open[key(c)]">
          {{ open[key(c)] ? "收合" : "展開全文" }}
        </button>
      </article>
    </TransitionGroup>
  </section>
</template>

<style scoped>
.cites { display: grid; gap: 10px; align-content: start; }
header { display: flex; align-items: center; gap: 8px; }
.count {
  font-size: 12px; font-weight: 600; min-width: 20px; text-align: center;
  padding: 0 6px; border-radius: 999px; background: var(--surface-2); color: var(--muted);
}
.note { margin: 0; font-size: 13px; color: var(--faint); }
.list { display: grid; gap: 10px; }
.cite { padding: 14px 16px; display: grid; gap: 8px; }
.top { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 10px; }
.no {
  font-size: 12px; font-weight: 700; color: var(--accent); background: var(--accent-soft);
  padding: 1px 8px; border-radius: 6px; font-variant-numeric: tabular-nums;
}
.top strong { font-size: 15px; }
.src { margin-left: auto; font-size: 12px; color: var(--faint); }
.sim { display: flex; align-items: center; gap: 8px; }
.sim span { height: 4px; border-radius: 2px; background: var(--accent); opacity: .7; }
.sim small { font-size: 11.5px; color: var(--faint); font-variant-numeric: tabular-nums; }
.text {
  margin: 0; font-size: 14px; line-height: 1.75; color: var(--muted); white-space: pre-wrap;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
.open .text { display: block; }
.more {
  justify-self: start; border: 0; background: none; padding: 0;
  font-size: 13px; color: var(--accent); font-weight: 600;
}

.rise-enter-active { transition: all .25s ease; }
.rise-enter-from { opacity: 0; transform: translateY(6px); }
</style>
