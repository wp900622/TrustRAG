# syntax=docker/dockerfile:1
#
# TrustRAG 問答服務的映像。
#
# 兩段式建置：builder 裝套件，runtime 只拿裝好的東西，
# 這樣編譯用的工具鏈不會留在最後的映像裡。
#
# 建置：
#     docker build -t trustrag-api .
# 想要含 OCR（掃描 PDF 才需要，會多約 250 MB）：
#     docker build --build-arg WITH_OCR=1 -t trustrag-api .

ARG PYTHON_VERSION=3.13

# ---------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim AS builder

ARG WITH_OCR=0
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# chromadb 與 PyMuPDF 在某些平台需要編譯，build-essential 只留在這一段
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install
COPY requirements.txt .

# 用真正的 venv，不要 pip install --prefix。
# --prefix 只是把檔案放到那個目錄，沒有任何東西告訴直譯器要去那裡找，
# 結果是 /opt/venv/bin/uvicorn 存在但 import uvicorn 失敗。
# venv 會產生 pyvenv.cfg 與自己的 bin/python，sys.path 才接得起來。
RUN python -m venv /opt/venv

# pytest 不進正式映像；OCR 那包用 build arg 決定
RUN grep -viE '^(pytest|rapidocr-onnxruntime)' requirements.txt > runtime.txt \
    && /opt/venv/bin/pip install -r runtime.txt redis \
    && if [ "$WITH_OCR" = "1" ]; then \
         /opt/venv/bin/pip install rapidocr-onnxruntime; \
       fi

# ---------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH=/opt/venv/bin:$PATH

COPY --from=builder /opt/venv /opt/venv

# 不要用 root 跑。UID 固定，好對應掛進來的 volume 權限
RUN useradd --create-home --uid 10001 trustrag
WORKDIR /app

# 只放服務真的需要的東西。這份清單是掃 import 算出來的，不是猜的：
#   python scripts 掃 day24_service/*.py 的 import 閉包 → 9 個本地模組
# 快取檔先進來，第一次啟動建索引就不必打 embedding API
COPY work_rules.md rubric_day18.json ./
COPY embeddings_cache.json chat_cache.json agent_cache_day21.json ./
COPY rag_core.py chunkers.py chroma_store.py prompts.py pipeline.py ./
COPY judges.py agent_day20.py agent_day21.py ingest_day22.py ./
COPY day24_service ./day24_service

# 這兩個目錄是狀態，正式跑的時候會被 volume 蓋掉；
# 先建好並給對擁有者，免得 volume 是空的時候寫不進去。
# chroma_db 的位置寫死在 chroma_store.py（那是前 23 天的證據，不能改），
# 所以 volume 只能掛在 /app/chroma_db，不能掛去別處再用環境變數指
RUN mkdir -p /app/chroma_db /data/jobs && chown -R trustrag:trustrag /app /data

USER trustrag

ENV DAY24_DOCUMENTS_ROOT=/app \
    DAY24_JOB_DB=/data/jobs/day24_jobs.sqlite3

EXPOSE 8024

# /readyz 要等索引建好才會回 204，所以 start-period 給足
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8024/readyz', timeout=4).status==204 else 1)"

# workers 由 compose 傳進來；預設 1，因為索引每個 worker 各建一份
CMD ["uvicorn", "day24_service.main:app", "--host", "0.0.0.0", "--port", "8024"]
