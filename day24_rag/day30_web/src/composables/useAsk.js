import { onBeforeUnmount, reactive } from "vue";
import { ApiError, ask, askStream } from "../api.js";

// 一次問答的全部狀態。元件只負責畫，不碰 fetch。
export function useAsk() {
  const state = reactive({
    question: "",
    mode: "pipeline",
    phase: "idle",          // idle → searching → answering → done
    answer: "",
    hasAnswer: false,       // answer 可以是 null（Day 28 的金額／時間上限），跟空字串不同
    citations: [],
    timing: {},             // { citationsMs, firstTokenMs, totalMs }
    trace: null,            // agent：{ searches, queries }
    stopped: null,          // { reason, searches }
    error: null,            // { code, message, requestId }
  });

  let controller = null;

  function reset(question, mode) {
    Object.assign(state, { question, mode, phase: "searching", answer: "", hasAnswer: false,
                           citations: [], timing: {}, trace: null, stopped: null, error: null });
  }

  // 問下一題、或離開頁面時，前一題的串流要真的斷掉。
  // 不 abort 的話畫面換了，連線還在，服務那邊照樣把答案生完、照樣付錢（Day 26）。
  // 只有串流那條路收得到：它是 async 產生器，斷線會變成 aclose()。
  // agent 那條是同步 handler，abort 只是前端不再等，服務端還是會跑完
  // （停得住它的是 Day 28 的上限，不是這裡）
  function cancel() {
    controller?.abort();
    controller = null;
  }
  onBeforeUnmount(cancel);

  async function run({ question, agent, maxSteps }) {
    cancel();
    reset(question, agent ? "agent" : "pipeline");
    controller = new AbortController();
    const { signal } = controller;
    try {
      if (agent) await runAgent(question, maxSteps, signal);
      else await runStream(question, signal);
    } catch (e) {
      if (e.name === "AbortError") return;           // 自己取消的，不是錯誤
      state.error = e instanceof ApiError
        ? { code: e.code, message: e.message, requestId: e.requestId }
        : { code: "network", message: `連不上服務：${e.message}` };
    } finally {
      if (!signal.aborted) state.phase = "done";
    }
  }

  async function runStream(question, signal) {
    for await (const { event, data } of askStream({ question }, signal)) {
      if (event === "citations") {
        // 出處先畫，這一刻模型還沒開口
        state.citations = data.citations;
        state.timing.citationsMs = data.elapsed_ms;
        state.phase = "answering";
      } else if (event === "token") {
        state.answer += data.text;
        state.hasAnswer = true;
      } else if (event === "error") {
        state.error = { code: data.code, message: data.message, requestId: data.request_id };
      } else if (event === "done") {
        state.timing.firstTokenMs = data.first_token_ms;
        state.timing.totalMs = data.timing.total_ms;
      }
    }
  }

  async function runAgent(question, maxSteps, signal) {
    const body = await ask({ question, mode: "agent", max_steps: maxSteps }, signal);
    const a = body.agent;
    state.citations = body.citations;
    state.hasAnswer = body.answer != null;
    state.answer = body.answer ?? "";
    state.trace = { searches: a.searches, queries: a.queries };
    state.timing.totalMs = body.timing.total_ms;
    // 被攔下來是 200 不是錯誤。不顯示 stopped_at_step：上限 2 時它是 3
    if (a.stopped_reason) state.stopped = { reason: a.stopped_reason, searches: a.searches };
  }

  return { state, run, cancel };
}
