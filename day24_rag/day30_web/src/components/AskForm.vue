<script setup>
import { ref } from "vue";

const props = defineProps({ busy: Boolean });
const emit = defineEmits(["ask"]);

const EXAMPLES = [
  "病假連續請幾天以上需要附診斷證明？",
  "忘記打卡要怎麼補救？會有事嗎？",
  "公司有健身房或運動補助嗎？",
];

const question = ref("");
const agent = ref(false);
const maxSteps = ref(4);

function submit() {
  const q = question.value.trim();
  if (q && !props.busy) emit("ask", { question: q, agent: agent.value, maxSteps: maxSteps.value });
}

function pick(q) {
  question.value = q;
  submit();
}

// Enter 送出，Shift+Enter 換行
function onKey(e) {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    submit();
  }
}
</script>

<template>
  <form class="card ask" @submit.prevent="submit">
    <div class="box">
      <textarea v-model="question" rows="2" maxlength="500" required
                placeholder="問一個工作規則的問題，例如：特休沒休完會怎樣？"
                @keydown="onKey" />
      <button class="send" :disabled="busy || !question.trim()" aria-label="送出">
        <svg v-if="!busy" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path d="M4 12h14M12 5l7 7-7 7" fill="none" stroke="currentColor" stroke-width="2.2"
                stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        <span v-else class="spinner" />
      </button>
    </div>

    <div class="controls">
      <div class="seg" role="radiogroup" aria-label="答法">
        <button type="button" role="radio" :aria-checked="!agent" :class="{ on: !agent }"
                @click="agent = false">
          快速回答 <small>查一次、答一次</small>
        </button>
        <button type="button" role="radio" :aria-checked="agent" :class="{ on: agent }"
                @click="agent = true">
          Agent <small>自己決定查幾次</small>
        </button>
      </div>
      <label v-if="agent" class="steps">
        最多 <b>{{ maxSteps }}</b> 步
        <input v-model.number="maxSteps" type="range" min="1" max="6" aria-label="最多幾步">
      </label>
    </div>

    <div class="examples">
      <span class="label">試試看</span>
      <button v-for="q in EXAMPLES" :key="q" type="button" class="chip" :disabled="busy"
              @click="pick(q)">{{ q }}</button>
    </div>
  </form>
</template>

<style scoped>
.ask { padding: 16px; display: grid; gap: 14px; }
.box {
  display: flex; align-items: flex-end; gap: 8px;
  border: 1px solid var(--line); border-radius: 10px; background: var(--bg);
  padding: 8px 8px 8px 14px; transition: border-color .15s;
}
.box:focus-within { border-color: var(--accent); }
textarea {
  flex: 1; border: 0; outline: 0; resize: none; background: transparent;
  font-size: 16px; line-height: 1.6; padding: 4px 0; min-height: 3.2em;
}
.send {
  flex: none; width: 40px; height: 40px; border-radius: 9px; border: 0;
  display: grid; place-items: center;
  background: var(--accent); color: var(--accent-fg); transition: opacity .15s;
}
.send:disabled { opacity: .4; }
.spinner {
  width: 16px; height: 16px; border-radius: 50%;
  border: 2px solid currentColor; border-right-color: transparent;
  animation: spin .7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px; }
.seg {
  display: inline-flex; padding: 3px; border-radius: 10px; background: var(--surface-2);
}
.seg button {
  border: 0; background: transparent; border-radius: 8px; padding: 6px 14px;
  font-weight: 600; font-size: 14px; color: var(--muted);
}
.seg button small { font-weight: 400; color: var(--faint); margin-left: 4px; }
.seg button.on { background: var(--surface); color: var(--fg); box-shadow: var(--shadow); }
.steps { display: inline-flex; align-items: center; gap: 8px; font-size: 14px; color: var(--muted); }
.steps b { color: var(--fg); font-variant-numeric: tabular-nums; }
.steps input { accent-color: var(--accent); width: 120px; }

.examples { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.chip {
  border: 1px solid var(--line); background: var(--surface); color: var(--muted);
  border-radius: 999px; padding: 4px 12px; font-size: 13px; transition: all .15s;
}
.chip:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }

@media (max-width: 560px) {
  .seg button small { display: none; }
}
</style>
