# -*- coding: utf-8 -*-
"""把腳本時代的檔案快取換成服務能用的版本。

`rag_core` 的快取是「整包讀進來、改一個 key、整包寫回去」。
單執行緒的腳本跑了 23 天沒出過事，但服務有兩條執行緒會同時走到那一行：

- 兩邊都拿著舊的那包，後寫的把先寫的蓋掉 → 靜靜掉資料
- `write_text()` 會先把檔案截成 0 再寫，另一條執行緒剛好在那個瞬間讀，
  會拿到半個檔案（UnicodeDecodeError）或空檔（JSONDecodeError）→ 整個請求 500

這支檔案修掉這兩件事，而且**不改 `rag_core.py`**（它是前 23 天每一篇的證據）：
啟動時把 `_load_json` / `_save_json` 換成下面這兩個。

三個保證：
1. 檔案只在啟動時讀一次，之後留在記憶體 → 請求路徑上沒有 10.5 MB 的讀檔
2. 讀寫共用一把鎖，而且回傳的是同一個 dict 物件 → 不會有人拿著舊的那包
3. 寫檔是「先寫暫存檔再 os.replace」 → 讀的人只會看到換之前或換之後
"""
import json
import os
import threading
from pathlib import Path

import rag_core

_LOCK = threading.RLock()
_MEMORY: dict[str, dict] = {}
_ORIGINAL: dict[str, object] = {}


def _load(path: Path) -> dict:
    key = str(path)
    with _LOCK:
        if key not in _MEMORY:
            _MEMORY[key] = (json.loads(path.read_text(encoding="utf-8"))
                            if path.exists() else {})
        return _MEMORY[key]


def _save(path: Path, data: dict) -> None:
    key = str(path)
    with _LOCK:
        # 呼叫端拿到的就是記憶體那一份，所以這裡多半是同一個物件；
        # 不是的話就合併進去，不要整包覆蓋掉別人剛寫的
        current = _MEMORY.setdefault(key, {})
        if current is not data:
            current.update(data)
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)


def install() -> None:
    """在 lifespan 裡呼叫一次。可重入，重複呼叫不會把原版蓋掉。"""
    with _LOCK:
        if _ORIGINAL:
            return
        _ORIGINAL["load"] = rag_core._load_json
        _ORIGINAL["save"] = rag_core._save_json
        rag_core._load_json = _load
        rag_core._save_json = _save


def uninstall() -> None:
    """還原。測試用，以及關機時讓行程乾淨退出。"""
    with _LOCK:
        if not _ORIGINAL:
            return
        rag_core._load_json = _ORIGINAL.pop("load")
        rag_core._save_json = _ORIGINAL.pop("save")
        _MEMORY.clear()


def warm(*paths: Path) -> dict[str, int]:
    """啟動時先把快取讀進記憶體，不要讓第一個使用者付這筆錢。"""
    return {p.name: len(_load(p)) for p in paths}


def stats() -> dict[str, int]:
    with _LOCK:
        return {Path(k).name: len(v) for k, v in _MEMORY.items()}
