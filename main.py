import os
import re
import json
import threading
from bisect import bisect_right
from collections import defaultdict

import duckdb
import gradio as gr
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from cachetools import TTLCache
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# ── CONFIG ──────────────────────────────────────────────────────────────────
SOURCE_REPO = os.environ.get("SOURCE_REPO", "mukeshcmdbrowser/icrm-hitek-fulldb")
INDEX_REPO = os.environ.get("INDEX_REPO", "Bhatiasab/icmr-phone-locator-private")
LEADING_DIGITS = os.environ.get("LEADING_DIGITS", "9")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN missing. Add it in Render → Environment.")

api = HfApi(token=HF_TOKEN)

def discover_source_files():
    try:
        files = api.list_repo_files(repo_id=SOURCE_REPO, repo_type="dataset")
        parquets = sorted(f for f in files if f.endswith(".parquet"))
        return parquets or ["part1.parquet", "part2a.parquet", "part2b_new.parquet"]
    except Exception as e:
        print(f"[warn] discovery failed: {e}")
        return ["part1.parquet", "part2a.parquet", "part2b_new.parquet"]

SOURCE_FILES = discover_source_files()
print(f"[init] source files: {SOURCE_FILES}")

COLUMNS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]

# ── STATE ───────────────────────────────────────────────────────────────────
STATE_LOCK = threading.Lock()
STATE = {
    "rowgroups_loaded": False,
    "indexed_digits": set(),
}
ROWGROUP_META = None
META_BY_ID = {}

def remote_exists(filename: str) -> bool:
    try:
        return api.file_exists(repo_id=INDEX_REPO, filename=filename, repo_type="dataset")
    except Exception:
        return False

def refresh_index_state():
    global ROWGROUP_META, META_BY_ID
    indexed = {d for d in "6789" if remote_exists(f"done/{d}.txt")}
    with STATE_LOCK:
        STATE["indexed_digits"] = indexed

    if not remote_exists("rowgroups.json"):
        return False

    if ROWGROUP_META is None:
        path = hf_hub_download(
            repo_id=INDEX_REPO, filename="rowgroups.json",
            repo_type="dataset", token=HF_TOKEN,
        )
        with open(path, "r", encoding="utf-8") as f:
            ROWGROUP_META = json.load(f)
        META_BY_ID = {int(item["id"]): item for item in ROWGROUP_META["files"]}

    with STATE_LOCK:
        STATE["rowgroups_loaded"] = True
    return True

# ── DUCKDB (remote-only, minimal memory) ────────────────────────────────────
hffs = HfFileSystem(token=HF_TOKEN, block_size=8 * 1024 * 1024)
db = duckdb.connect()
db.execute("INSTALL httpfs; LOAD httpfs;")
db.execute("SET threads = 1;")            # Render free tier safe
db.execute("SET memory_limit = '400MB';") # Render 512MB safe
db.execute("SET parquet_metadata_cache = true;")
safe_token = HF_TOKEN.replace("'", "''")
db.execute(f"CREATE OR REPLACE SECRET hf_auth (TYPE huggingface, TOKEN '{safe_token}');")

DB_LOCK = threading.Lock()
RAW_LOCK = threading.Lock()
RAW_HANDLES = {}
RAW_PARQUETS = {}

# ── HELPERS ─────────────────────────────────────────────────────────────────
def normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    elif len(digits) > 10:
        digits = digits[-10:]
    if len(digits) != 10:
        raise ValueError("Phone must have exactly 10 digits.")
    if digits[0] not in "6789":
        raise ValueError("Unsupported Indian mobile prefix.")
    return digits

def get_raw_parquet(file_id: int):
    with RAW_LOCK:
        if file_id in RAW_PARQUETS:
            return RAW_PARQUETS[file_id]
        remote = f"datasets/{SOURCE_REPO}/{SOURCE_FILES[file_id]}"
        handle = hffs.open(remote, "rb")
        parquet = pq.ParquetFile(handle)
        RAW_HANDLES[file_id] = handle
        RAW_PARQUETS[file_id] = parquet
        return parquet

def find_locations(phone: str):
    lead, p4 = phone[0], phone[:4]
    path = f"hf://datasets/{INDEX_REPO}/index/lead={lead}/p4={p4}/*.parquet"
    safe_path = path.replace("'", "''")
    sql = f"SELECT file_id, row_num FROM read_parquet('{safe_path}', hive_partitioning = false) WHERE phoneNumber = ?"
    with DB_LOCK:
        rows = db.execute(sql, [phone]).fetchall()
    return [(int(fid), int(rn)) for fid, rn in rows]

def locate_rowgroup(file_id: int, row_num: int):
    meta = META_BY_ID[file_id]
    starts = meta["row_group_starts"]
    rg_id = bisect_right(starts, row_num) - 1
    if rg_id < 0:
        raise RuntimeError("Invalid row locator.")
    return rg_id, row_num - starts[rg_id]

def fetch_original_rows(locations, wanted_phone):
    grouped = defaultdict(list)
    for file_id, row_num in locations:
        rg_id, offset = locate_rowgroup(file_id, row_num)
        grouped[(file_id, rg_id)].append(offset)

    results = []
    for (file_id, rg_id), offsets in grouped.items():
        parquet = get_raw_parquet(file_id)
        with RAW_LOCK:
            table = parquet.read_row_group(rg_id, columns=COLUMNS, use_threads=True)
        for offset in offsets:
            if offset >= table.num_rows:
                continue
            record = table.slice(offset, 1).to_pylist()[0]
            if str(record.get("phoneNumber")) != wanted_phone:
                continue
            results.append(record)
    return results

def mask_aadhaar(value):
    if value is None: return None
    d = re.sub(r"\D", "", str(value))
    return "XXXXXXXX" + d[-4:] if len(d) >= 4 else None

def mask_other(value):
    if value is None: return None
    d = re.sub(r"\D", "", str(value))
    return "XXXXXX" + d[-4:] if len(d) >= 4 else None

def sanitize(r):
    return {
        "name": r.get("name"),
        "fathersName": "[REDACTED]" if r.get("fathersName") else None,
        "phoneNumber": r.get("phoneNumber"),
        "aadharNumber": mask_aadhaar(r.get("aadharNumber")),
        "otherNumber": mask_other(r.get("otherNumber")),
        "address": "[REDACTED]" if r.get("address") else None,
        "district": r.get("district"),
        "pincode": r.get("pincode"),
        "state": r.get("state"),
        "town": r.get("town"),
        "source": r.get("source"),
    }

_cache = TTLCache(maxsize=2000, ttl=3600)
_cache_lock = threading.Lock()

def cached_lookup(phone: str):
    with _cache_lock:
        if phone in _cache:
            return _cache[phone]
    locations = find_locations(phone)
    if not locations:
        result = tuple()
    else:
        records = fetch_original_rows(locations, phone)
        result = tuple(json.dumps(sanitize(r), ensure_ascii=False) for r in records)
    with _cache_lock:
        _cache[phone] = result
    return result

# ── PUBLIC API FUNCTIONS ────────────────────────────────────────────────────
def lookup_api(number: str) -> dict:
    try:
        phone = normalize_phone(number)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "results": []}
    try:
        values = cached_lookup(phone)
        results = [json.loads(v) for v in values]
        return {"ok": True, "query": phone, "count": len(results), "results": results}
    except Exception as exc:
        return {"ok": False, "query": phone, "error": str(exc), "results": []}

def status_api() -> dict:
    try:
        refresh_index_state()
    except Exception:
        pass
    with STATE_LOCK:
        return {
            "status": "ok",
            "source_repo": SOURCE_REPO,
            "index_repo": INDEX_REPO,
            "source_files": SOURCE_FILES,
            "indexed_digits": sorted(STATE["indexed_digits"]),
            "rowgroups_loaded": STATE["rowgroups_loaded"],
        }

# ── GRADIO UI ───────────────────────────────────────────────────────────────
with gr.Blocks(title="Private Phone Lookup") as demo:
    gr.Markdown("""
# 🔍 Private Phone Lookup

CPU-only lookup. No GPU used.

Enter a 10-digit Indian mobile number to search.
""")
    with gr.Row():
        number_input = gr.Textbox(label="Phone Number", placeholder="e.g. 9876543210")
        lookup_button = gr.Button("Lookup", variant="primary")

    result_output = gr.JSON(label="Result")
    status_button = gr.Button("Build Status")
    status_output = gr.JSON(label="Status")

    lookup_button.click(fn=lookup_api, inputs=number_input, outputs=result_output, api_name="lookup")
    number_input.submit(fn=lookup_api, inputs=number_input, outputs=result_output)
    status_button.click(fn=status_api, inputs=None, outputs=status_output, api_name="build_status")
    gr.api(lookup_api, api_name="lookup_json")
    gr.api(status_api, api_name="status_json")

# ── FASTAPI WRAPPER (Render health check) ───────────────────────────────────
fastapi_app = FastAPI(title="Phone Lookup API")

@fastapi_app.get("/health")
def health():
    return {"status": "ok", "source_repo": SOURCE_REPO}

@fastapi_app.get("/ready")
def ready():
    with STATE_LOCK:
        return {
            "indexed_digits": sorted(STATE["indexed_digits"]),
            "rowgroups_loaded": STATE["rowgroups_loaded"],
        }

# Mount Gradio at /
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# ── STARTUP ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[init] checking index state...")
    try:
        refresh_index_state()
        print(f"[init] indexed digits: {sorted(STATE['indexed_digits'])}")
    except Exception as e:
        print(f"[init] index check failed: {e!r}")

    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        show_api=False,
    )
