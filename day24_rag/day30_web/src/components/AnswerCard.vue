<script setup>
import { computed } from "vue";

const props = defineProps({ state: { type: Object, required: true } });

const STOPPED = {
  steps: "查詢次數到了上限，已用手上的資料直接作答，答案可能不完整。",
  cost: "花費到了上限，這一題沒有答案。右邊是停下來之前找到的條文，不代表規章裡沒有。",
  time: "時間到了上限，這一題沒有答案。右邊是停下來之前找到的條文，不代表規章裡沒有。",
};

const ms = (v) => (v == null ? null : v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(1)} s`);
const busy = computed(() => ["searching", "answering"].includes(props.state.phase));
const waitingText = computed(() =>
  props.state.mode === "agent" ? "Agent 正在查規章，查完才會一次給答案…" : "正在找相關條文…");
</script>

<template>
  <section class="card answer">
    <header>
      <span class="label">答案</span>
      <span class="mode">{{ state.mode === "agent" ? "Agent" : "快速回答" }}</span>
    </header>

    <p class="q">{{ state.question }}</p>

    <div v-if="state.error" class="banner danger" role="alert">
      <b>{{ state.error.code }}</b>　{{ state.error.message }}
      <small v-if="state.error.requestId">request_id {{ state.error.requestId }}</small>
    </div>

    <div v-if="state.stopped" class="banner warn" role="status">
      <b>被攔下來，查了 {{ state.stopped.searches }} 次</b>
      {{ STOPPED[state.stopped.reason] ?? state.stopped.reason }}
    </div>

    <div class="body">
      <p v-if="state.hasAnswer" class="text">{{ state.answer }}<span v-if="busy" class="caret" /></p>
      <p v-else-if="busy" class="wait">
        <span class="dots"><i /><i /><i /></span>{{ waitingText }}
      </p>
      <p v-else-if="state.stopped" class="none">（沒有答案）</p>
    </div>

    <footer v-if="state.timing.totalMs != null || state.trace">
      <div v-if="state.trace" class="queries">
        <span class="label">查過</span>
        <span v-for="(q, i) in state.trace.queries" :key="i" class="pill">{{ q }}</span>
      </div>
      <div class="timing">
        <span v-if="state.timing.citationsMs != null" class="pill">出處 <b>{{ ms(state.timing.citationsMs) }}</b></span>
        <span v-if="state.timing.firstTokenMs != null" class="pill">第一個字 <b>{{ ms(state.timing.firstTokenMs) }}</b></span>
        <span v-if="state.timing.totalMs != null" class="pill">完成 <b>{{ ms(state.timing.totalMs) }}</b></span>
      </div>
    </footer>
  </section>
</template>

<style scoped>
.answer { padding: 18px 20px; display: grid; gap: 12px; }
header { display: flex; align-items: center; justify-content: space-between; }
.mode { font-size: 12px; color: var(--accent); background: var(--accent-soft); padding: 2px 10px; border-radius: 999px; font-weight: 600; }
.q { margin: 0; font-size: 17px; font-weight: 600; line-height: 1.5; }
.body { min-height: 3.2em; }
.text { margin: 0; font-size: 16px; line-height: 1.8; white-space: pre-wrap; }
.wait, .none { margin: 0; color: var(--muted); display: flex; align-items: center; gap: 10px; }
.caret {
  display: inline-block; width: 2px; height: 1.1em; margin-left: 2px; vertical-align: text-bottom;
  background: var(--accent); animation: blink 1s steps(1) infinite;
}
@keyframes blink { 50% { opacity: 0; } }
.dots { display: inline-flex; gap: 4px; }
.dots i {
  width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
  animation: bounce 1s ease-in-out infinite;
}
.dots i:nth-child(2) { animation-delay: .15s; }
.dots i:nth-child(3) { animation-delay: .3s; }
@keyframes bounce { 0%, 100% { opacity: .25; } 50% { opacity: 1; } }

.banner { border-radius: 10px; padding: 10px 14px; font-size: 14px; line-height: 1.6; border: 1px solid; }
.banner b { margin-right: 6px; }
.banner small { display: block; opacity: .75; margin-top: 2px; }
.warn { background: var(--warn-soft); color: var(--warn); border-color: var(--warn-line); }
.danger { background: var(--danger-soft); color: var(--danger); border-color: transparent; }

footer { display: grid; gap: 10px; padding-top: 12px; border-top: 1px dashed var(--line); }
.queries, .timing { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.queries .label { margin-right: 4px; }
</style>
