// 跟服務講話的地方，只有這一個檔案知道 URL 與線路格式。
// 形狀照 day24_service/contracts.py：AskRequest 進、AskResponse 出，錯誤是 Day 25 那個
// {"error": {"code", "message", "request_id"}}。

export class ApiError extends Error {
  constructor({ code, message, request_id }, status) {
    super(message);
    this.code = code;
    this.requestId = request_id;
    this.status = status;
  }
}

async function raiseFor(r) {
  let body = {};
  try { body = await r.json(); } catch { /* 不是 JSON 就用狀態碼 */ }
  throw new ApiError(body.error || { code: `http_${r.status}`, message: r.statusText }, r.status);
}

function post(path, body, signal) {
  return fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

/** 非串流：agent 模式只能走這條（Day 26：它中間的 token 不是最終答案）。 */
export async function ask(body, signal) {
  const r = await post("/ask", body, signal);
  if (!r.ok) await raiseFor(r);
  return r.json();
}

/**
 * 串流：一個 async generator，一次吐一個 SSE 事件 {event, data}。
 * EventSource 只能發 GET，而 /ask 是 POST，所以自己讀 body、自己切 "\n\n"。
 */
export async function* askStream(body, signal) {
  const r = await post("/ask", { ...body, stream: true }, signal);
  if (!r.ok) await raiseFor(r);
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += value;
    let cut;
    while ((cut = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      const event = /^event: (.*)$/m.exec(block)?.[1];
      const data = /^data: (.*)$/m.exec(block)?.[1];
      if (event) yield { event, data: data ? JSON.parse(data) : {} };
    }
  }
}
