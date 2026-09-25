# -*- coding: utf-8 -*-
"""Day 30：替 /ui/（day30_web 的 Vue 版）拍兩張圖（pipeline 一張、agent 被攔下來一張），給文章用。

用系統內建的 Edge 以 headless 模式開頁面，透過 DevTools 協定填問題、送出、等答案，再截圖。
題目都在快取裡，不打 API。先開服務在 8024。
"""
import base64
import json
import subprocess
import tempfile
import time
import urllib.request

import websockets.sync.client as ws_client

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9230
URL = "http://127.0.0.1:8024/ui/"

# Vue 版：直接改 value，v-model 收不到，要補發 input／change 事件
SET = ("const set=(el,v)=>{if(el.type==='checkbox'){el.checked=v;el.dispatchEvent(new Event('change'))}"
       "else{el.value=v;el.dispatchEvent(new Event('input'))}};")
SUBMIT = "await new Promise(r=>setTimeout(r,50));document.querySelector('form').requestSubmit();"

TEXT = "document.querySelector('textarea')"
SHOTS = [
    ("day30_ui_pipeline.png",
     SET + f"set({TEXT},'病假連續請幾天以上需要附診斷證明？');" + SUBMIT),
    ("day30_ui_agent.png",
     SET + f"set({TEXT},'公司有健身房或運動補助嗎？');"
     "document.querySelectorAll('.seg button')[1].click();"
     "await new Promise(r=>setTimeout(r,80));"
     "set(document.querySelector('input[type=range]'),'2');" + SUBMIT),
]


class CDP:
    def __init__(self, url):
        self.ws = ws_client.connect(url, max_size=None)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                return msg.get("result", {})


def main():
    profile = tempfile.mkdtemp()
    proc = subprocess.Popen([EDGE, "--headless=new", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={profile}", "--window-size=1200,1000",
                             "--force-device-scale-factor=1", "about:blank"])
    try:
        for _ in range(50):
            try:
                pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                page = next(p for p in pages if p["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.call("Emulation.setDeviceMetricsOverride", width=1200, height=1000,
                 deviceScaleFactor=1, mobile=False)
        for out, fill in SHOTS:
            cdp.call("Page.navigate", url=URL)
            time.sleep(1.0)
            cdp.call("Runtime.evaluate", expression="(async()=>{" + fill + "})()",
                     awaitPromise=True)
            time.sleep(2.5)
            h = cdp.call("Runtime.evaluate",
                         expression="Math.ceil(document.body.getBoundingClientRect().bottom) + 16",
                         returnByValue=True)["result"]["value"]
            shot = cdp.call("Page.captureScreenshot", format="png",
                            captureBeyondViewport=True,
                            clip={"x": 0, "y": 0, "width": 1200,
                                  "height": min(h, 1600), "scale": 1})
            with open(out, "wb") as f:
                f.write(base64.b64decode(shot["data"]))
            print(out, h)
    finally:
        # terminate() 只關主行程，Edge 的子行程會留著占住除錯埠，下一次就連不上
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True)


if __name__ == "__main__":
    main()
