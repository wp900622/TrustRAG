<script setup>
import AnswerCard from "./components/AnswerCard.vue";
import AskForm from "./components/AskForm.vue";
import CitationList from "./components/CitationList.vue";
import { useAsk } from "./composables/useAsk.js";
import { useHealth } from "./composables/useHealth.js";

const { state, run } = useAsk();
const health = useHealth();

const STATUS = { checking: "連線中", ok: "服務正常", warming: "索引建立中", down: "連不上服務" };
</script>

<template>
  <div class="shell">
    <header class="top">
      <div class="brand">
        <span class="logo" aria-hidden="true">規</span>
        <div>
          <h1>規章問答</h1>
          <p>TrustRAG・公司工作規則</p>
        </div>
      </div>
      <span class="status" :class="health.status">
        <i />{{ STATUS[health.status] }}
        <template v-if="health.status === 'ok'">・{{ health.chunks }} 段條文</template>
      </span>
    </header>

    <AskForm :busy="state.phase === 'searching' || state.phase === 'answering'" @ask="run" />

    <main v-if="state.phase !== 'idle'" class="result">
      <AnswerCard :state="state" />
      <CitationList v-if="state.citations.length" :items="state.citations" :mode="state.mode" />
    </main>

    <section v-else class="empty">
      <p>答案會附上它查到的條文。<br>「快速回答」查一次就答；「Agent」會自己換關鍵字多查幾次，比較慢。</p>
    </section>
  </div>
</template>

<style scoped>
.shell { max-width: 1120px; margin: 0 auto; padding: 20px 16px 48px; display: grid; gap: 20px; }
.top { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.brand { display: flex; align-items: center; gap: 12px; }
.logo {
  width: 40px; height: 40px; border-radius: 10px; display: grid; place-items: center;
  background: var(--accent); color: var(--accent-fg); font-weight: 700; font-size: 18px;
}
h1 { margin: 0; font-size: 20px; line-height: 1.2; }
.brand p { margin: 2px 0 0; font-size: 13px; color: var(--faint); }
.status {
  display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted);
  padding: 4px 12px; border-radius: 999px; background: var(--surface); border: 1px solid var(--line);
}
.status i { width: 8px; height: 8px; border-radius: 50%; background: var(--faint); }
.status.ok i { background: #22c55e; }
.status.warming i { background: #f59e0b; }
.status.down i { background: #ef4444; }

.result { display: grid; gap: 20px; align-items: start; }
@media (min-width: 960px) {
  .result { grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); }
  .result > :first-child { position: sticky; top: 20px; }
}
.empty {
  text-align: center; color: var(--faint); font-size: 14px; line-height: 1.9;
  padding: 40px 16px; border: 1px dashed var(--line); border-radius: var(--radius);
}
.empty p { margin: 0; }
</style>
