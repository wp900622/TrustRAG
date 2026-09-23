# 部署：本機 Docker → Google Cloud

本機與雲端跑的是同一個映像、同一組服務，差別只在映像是現場 build 還是用拉的。

```
Dockerfile              兩段式建置：builder 裝套件，runtime 只拿裝好的
docker-compose.yml      本機用：api ＋ redis 兩個服務，三個具名 volume
docker-compose.gce.yml  雲端用：一樣的東西，但映像用拉的不是現場 build
.dockerignore           把 11 MB 的快取與一堆圖擋在 build context 外
.env.docker.example     複製成 .env 再填
documents/              要攝取的文件放這，唯讀掛進容器
```

---

## 一、本機跑起來

```bash
cp .env.docker.example .env      # 填 OPENAI_API_KEY
docker compose up -d --build
```

第一次 build 要幾分鐘（`chromadb` 有東西要編譯），之後有 layer cache 就快了。

確認：

```bash
curl -s localhost:8024/healthz | python -m json.tool
```

要看到 `"ready": true`、`"chunks": 59`，以及 `"cache": {"backend": "redis", ...}`。
`backend` 如果是 `memory`，代表 api 連不到 redis，看 `docker compose logs api | grep -i redis`。

問一題：

```bash
curl -s -X POST localhost:8024/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"病假請幾天要附診斷證明？","k":3}'
```

### 三個 volume 各自裝什麼

| volume | 掛在哪 | 裝什麼 | 砍掉會怎樣 |
| --- | --- | --- | --- |
| `chroma-data` | `/app/chroma_db` | 向量庫 | 下次啟動重建，約幾秒，不花錢（向量在快取裡） |
| `job-data` | `/data/jobs` | 攝取 job 的 SQLite | 歷史 job 紀錄不見，服務照跑 |
| `redis-data` | `/data` | 問題向量與答案的快取 | 下次啟動用映像裡的 JSON 重新灌，只掉之後新增的部分 |

`chroma_db` 的位置**寫死在 `chroma_store.py`**，所以 volume 只能掛 `/app/chroma_db`，
不能掛別的地方再用環境變數指過去。那個檔案是前 23 天每一篇的證據，不改。

### Redis 裡面長什麼樣

```bash
docker compose exec redis redis-cli DBSIZE
docker compose exec redis redis-cli --scan --pattern 'day24:*' | head
docker compose exec redis redis-cli INFO memory | grep used_memory_human
```

key 長這樣：`day24:embeddings_cache:<16 碼雜湊>`、`day24:chat_cache:<16 碼雜湊>`。
前綴可以用 `DAY24_REDIS_PREFIX` 換，多個環境共用一台 Redis 時會用到。

### 要把 Redis 裡的快取倒回檔案

版控裡那兩個 JSON 是讓讀者重跑拿到相同數字用的。跑久了 Redis 裡會比檔案多，
要更新版控時：

```bash
docker compose exec api python -c "
from day24_service import cache
import rag_core
cache.install()
print('embeddings:', cache.export(rag_core.CACHE_PATH))
print('chat      :', cache.export(rag_core.CHAT_CACHE_PATH))
"
docker compose cp api:/app/chat_cache.json ./chat_cache.json
```

### 本機驗過的結果

2026-09-20 在 Windows 11 ＋ Docker Desktop 上跑完的。映像 945 MB。

```
/healthz   ready=true  chunks=59  boot_ms=724.9
           cache={"backend":"redis","keys":1644,"used_memory_human":"3.73M"}
/readyz    204
```

同一題問四次（第四次是砍掉 api 容器重建之後問的）：

| 第幾次 | embed | llm | total | cached | Redis key 數 |
| --- | --- | --- | --- | --- | --- |
| 1（冷） | 3820.2 ms | 2213.4 ms | 6082.0 ms | false | 1644 → 1646 |
| 2 | 1.5 ms | 0.8 ms | 11.5 ms | true | 1646 |
| 3 | 0.8 ms | 0.6 ms | 11.2 ms | true | 1646 |
| 4（新容器） | 0.9 ms | 1.1 ms | 15.4 ms | true | 1646 |

四次答案的雜湊完全相同。第四次是重點：容器整個換掉，快取還在，
因為它不在容器裡。新容器的啟動 log 是

```
cache={'backend': 'redis', 'embeddings_cache.json': 0, 'chat_cache.json': 0}
```

那兩個 `0` 是 `seed()` 用 SETNX 寫進去的新 key 數——一把都沒寫，代表 Redis 裡本來就有。
啟動時間也從 724 ms 掉到 214 ms。

---

## 二、踩過的坑

### `pip install --prefix` 會讓 import 失敗

第一版 Dockerfile 用 `pip install --prefix=/opt/venv`，起來之後 api 一直重啟：

```
ModuleNotFoundError: No module named 'uvicorn'
  File "/opt/venv/bin/uvicorn", line 3, in <module>
```

執行檔在、模組不在。進映像裡看：

```
套件在   /opt/venv/lib/python3.13/site-packages
直譯器是 /usr/local/bin/python
sys.path 只有 /usr/local/lib/python3.13/site-packages
```

`--prefix` 只是把檔案**放**到那個目錄，沒有讓任何直譯器去那裡找。
`PATH=/opt/venv/bin:$PATH` 讓 shell 找得到 `uvicorn` 這個腳本，
但腳本的 shebang 指回系統 python，import 就掛了。
兩段式建置的教學這樣寫能跑，是因為他們另外設了 `PYTHONPATH`——這裡的 `PYTHONPATH` 被 `/app` 佔走。

改成 `RUN python -m venv /opt/venv` 再用 `/opt/venv/bin/pip` 裝就對了。
venv 會產生 `pyvenv.cfg` 和自己的 `bin/python`，`sys.path` 才接得起來。
`COPY --from=builder /opt/venv /opt/venv` 照樣可行，兩段用的是同一個 base image。

**改完先驗 import 再起服務**，不要等重啟迴圈才發現：

```bash
docker run --rm --entrypoint python trustrag-api:latest -c   "import uvicorn, fastapi, redis, chromadb; import day24_service.main; print('ok')"
```

### curl 送中文 body 會被 shell 弄壞

Git Bash 下直接 `-d '{"question":"病假…"}'` 會拿到
`{"detail":"There was an error parsing the body"}`，那是終端機的編碼問題不是服務的。
把 JSON 寫成 UTF-8 檔案再送：

```bash
curl -s -X POST localhost:8024/ask   -H 'Content-Type: application/json; charset=utf-8'   --data-binary @ask.json
```


---

## 三、開到 Google Cloud

用一台 Compute Engine VM 跑同一個映像。沒有用 Memorystore，
因為它只在 VPC 內，Cloud Run 要連得到得多開 Serverless VPC connector，
以這個規模來說那是多付一份錢買一層不需要的東西。

### 需要先有的

- 一個 GCP 專案，計費已開啟
- 本機裝好 `gcloud`（<https://cloud.google.com/sdk/docs/install>），`gcloud auth login`

```bash
export PROJECT_ID=你的專案id
export REGION=asia-east1          # 台灣
export ZONE=asia-east1-b
gcloud config set project $PROJECT_ID
gcloud services enable compute.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com
```

### 1. 把金鑰放進 Secret Manager

不要放在 VM 的檔案裡，也不要寫進 metadata。

```bash
printf 'sk-你的金鑰' | gcloud secrets create openai-api-key --data-file=-
```

### 2. 建映像並推上 Artifact Registry

```bash
gcloud artifacts repositories create trustrag \
  --repository-format=docker --location=$REGION

gcloud auth configure-docker ${REGION}-docker.pkg.dev

IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/trustrag/api:$(date +%Y%m%d-%H%M)
docker build -t $IMAGE .
docker push $IMAGE
echo $IMAGE          # 記下來，compose 要用
```

用 Artifact Registry 而不是在 VM 上 build：VM 開小台的時候編 `chromadb` 會很久，
而且每次部署都重編一次沒有意義。

### 3. 開一台 VM

e2-small（2 vCPU、2 GB）跑得動：Redis 上限 512 MB，索引 59 塊很小，
真正吃記憶體的是 Python 與 `chromadb`。語料變大再往上調。

```bash
gcloud compute instances create trustrag \
  --zone=$ZONE \
  --machine-type=e2-small \
  --image-family=cos-stable --image-project=cos-cloud \
  --boot-disk-size=20GB \
  --scopes=cloud-platform \
  --tags=trustrag-api
```

用 Container-Optimized OS：Docker 已經在裡面，不用自己裝，而且它會自動更新。

**開防火牆**。只開給你自己的 IP，這支服務沒有任何認證：

```bash
MYIP=$(curl -s ifconfig.me)
gcloud compute firewall-rules create trustrag-api \
  --allow=tcp:8024 --target-tags=trustrag-api --source-ranges=${MYIP}/32
```

要對外開放之前，先在前面擺一層有認證的東西。現在 `/ask` 是誰打都會回答，
而每一次回答都在花你的 OpenAI 額度。

### 4. 把東西送上去

```bash
gcloud compute scp docker-compose.gce.yml trustrag:~/docker-compose.yml --zone=$ZONE
gcloud compute ssh trustrag --zone=$ZONE
```

進去之後：

```bash
# 讓 docker 拉得動 Artifact Registry
docker-credential-gcr configure-docker --registries=asia-east1-docker.pkg.dev

mkdir -p ~/documents

# 金鑰從 Secret Manager 拿。.env 只有自己讀得到
cat > ~/.env <<EOF
IMAGE=貼上剛才 echo 出來的那串
OPENAI_API_KEY=$(gcloud secrets versions access latest --secret=openai-api-key)
API_PORT=8024
DAY24_WORKERS=1
REDIS_MAXMEMORY=512mb
EOF
chmod 600 ~/.env

docker compose up -d
docker compose ps
curl -s localhost:8024/healthz
```

`IMAGE` 沒設的話 compose 會直接報錯停下來，不會默默跑到一個舊版本。

COS 上沒有 `docker compose` 子命令的話，改用
`--image-family=ubuntu-2204-lts` 開機器，再 `apt install docker-compose-plugin`。

### 5. 確認外面連得到

```bash
IP=$(gcloud compute instances describe trustrag --zone=$ZONE \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
curl -s http://$IP:8024/healthz
```

---

## 四、之後要做的事

### 更新版本

```bash
IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/trustrag/api:$(date +%Y%m%d-%H%M)
docker build -t $IMAGE . && docker push $IMAGE
gcloud compute ssh trustrag --zone=$ZONE --command \
  "sed -i 's|image: .*trustrag/api:.*|image: $IMAGE|' docker-compose.yml && \
   docker compose up -d && docker compose ps"
```

用日期當 tag 不用 `latest`：出事的時候要回得去上一個版本。

### 看 log

```bash
docker compose logs -f api
docker compose logs --since 10m api | grep -E 'ERROR|WARNING'
```

每一行請求 log 都帶 `request_id`，跟回應 header 的 `x-request-id` 是同一個，
使用者回報問題時可以用它對。

### 備份

真正不能掉的只有 Redis 裡新增的快取（那是花錢算出來的）。

```bash
gcloud compute ssh trustrag --zone=$ZONE --command \
  "docker compose exec -T redis redis-cli BGSAVE && sleep 3 && \
   docker run --rm -v trustrag_redis-data:/d -v \$PWD:/out alpine \
     tar czf /out/redis-\$(date +%F).tgz -C /d ."
```

`chroma_db` 不用備份，它從 `work_rules.md` 加快取就能重建。

---

## 五、還沒做的

這份部署**能跑，但還不能叫上線**。缺的東西照重要性排：

| 缺什麼 | 現在的狀況 | 不補的後果 |
| --- | --- | --- |
| **認證** | `/ask` 誰打都會回答 | 有人掃到你的 IP，就開始用你的 OpenAI 額度 |
| **HTTPS** | 純 HTTP | 問題與答案在網路上是明文 |
| **費用上限** | 沒有 | 被打爆的時候帳單沒有天花板 |
| 每個 IP 的速率限制 | 沒有 | 同上 |
| 監控與告警 | 只有 `docker compose logs` | 掛了沒人知道 |

最小的補法：前面放一台 Caddy（自動處理 Let's Encrypt 憑證）＋ Basic Auth，
OpenAI 那邊在專案上設一個每月用量上限。這兩件事加起來大概一小時。
