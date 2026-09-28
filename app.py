import os
import time
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download
import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "sauravsingh2111/Tgdata"
CACHE_DIR = os.getenv("CACHE_DIR", "/data/tgdb_cache")

con = None
is_ready = False
init_error = None
stats = {"startup_time": 0, "total_rows": 0, "queries": 0, "files": 0}


def init_dataset():
    global con, is_ready, init_error

    try:
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Check if parquet files exist
        parquet_glob = os.path.join(CACHE_DIR, "**", "*.parquet")
        existing = []
        for root, _, files in os.walk(CACHE_DIR):
            existing += [f for f in files if f.endswith(".parquet")]

        if not existing:
            logger.info("Downloading dataset from HuggingFace...")
            t0 = time.time()
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=CACHE_DIR,
                allow_patterns=["*.parquet"],
            )
            logger.info(f"Download done in {time.time()-t0:.1f}s")

        # Count parquet files
        file_list = []
        for root, _, files in os.walk(CACHE_DIR):
            for f in files:
                if f.endswith(".parquet"):
                    file_list.append(os.path.join(root, f))

        logger.info(f"Found {len(file_list)} parquet files")

        # DuckDB connection — out-of-core, low RAM
        conn = duckdb.connect(database=":memory:")
        conn.execute("PRAGMA threads=4;")
        conn.execute("PRAGMA memory_limit='450MB';")   # free/Starter safe
        conn.execute("PRAGMA temp_directory='/tmp/duckdb_tmp';")

        # Create view over all parquet files (lazy, no load)
        conn.execute(f"""
            CREATE VIEW tg AS
            SELECT * FROM read_parquet('{CACHE_DIR}/**/*.parquet', union_by_name=true)
        """)

        # Quick row count (DuckDB metadata only)
        row_count = conn.execute("SELECT COUNT(*) FROM tg").fetchone()[0]

        stats["total_rows"] = row_count
        stats["files"] = len(file_list)
        stats["startup_time"] = time.time()

        con = conn
        is_ready = True
        logger.info(f"✅ Ready: {len(file_list)} files, {row_count:,} rows")

    except Exception as e:
        init_error = str(e)
        logger.error(f"❌ Init failed: {e}", exc_info=True)


async def background_init():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, init_dataset)


@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(background_init())
    yield


app = FastAPI(
    title="Telegram DB API",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_ready():
    if not is_ready:
        if init_error:
            raise HTTPException(503, f"Init error: {init_error}")
        raise HTTPException(503, "Dataset loading, retry in a moment")


ALL_COLS = ["user_id", "username", "first_name", "last_name",
            "phone", "email", "status", "linked_id", "linked_name", "linked_handle"]


@app.get("/")
async def root():
    return {
        "name": "Telegram DB API",
        "dataset": REPO_ID,
        "ready": is_ready,
        "endpoints": ["/user/{id}", "/search", "/health"],
    }


@app.get("/user/{user_id}")
async def get_user(user_id: int):
    require_ready()
    stats["queries"] += 1
    t0 = time.time()

    cols = ", ".join(ALL_COLS)
    row = con.execute(
        f"SELECT {cols} FROM tg WHERE user_id = ? LIMIT 1", [user_id]
    ).fetchone()

    elapsed = (time.time() - t0) * 1000
    if not row:
        raise HTTPException(404, f"User {user_id} not found")

    return {
        "found": True,
        "query_time_ms": round(elapsed, 2),
        "user": dict(zip(ALL_COLS, row)),
    }


@app.get("/search")
async def search(
    username: str = Query(None),
    phone: str = Query(None),
    email: str = Query(None),
    first_name: str = Query(None),
    last_name: str = Query(None),
    limit: int = Query(10, ge=1, le=100),
):
    require_ready()
    stats["queries"] += 1
    t0 = time.time()

    fields = {"username": username, "phone": phone, "email": email,
              "first_name": first_name, "last_name": last_name}
    active = {k: v for k, v in fields.items() if v}

    if not active:
        raise HTTPException(400, "Provide at least one search parameter")

    where = " AND ".join([f"{k} = ?" for k in active])
    params = list(active.values())

    cols = ["user_id", "username", "first_name", "last_name",
            "phone", "email", "status"]

    rows = con.execute(
        f"SELECT {', '.join(cols)} FROM tg WHERE {where} LIMIT ?",
        params + [limit],
    ).fetchall()

    elapsed = (time.time() - t0) * 1000
    if not rows:
        raise HTTPException(404, "No users found")

    return {
        "found": True,
        "returned": len(rows),
        "query_time_ms": round(elapsed, 2),
        "users": [dict(zip(cols, r)) for r in rows],
    }


@app.get("/health")
async def health():
    uptime = round(time.time() - stats["startup_time"], 1) if stats["startup_time"] else 0
    return {
        "status": "ok" if is_ready else ("error" if init_error else "loading"),
        "ready": is_ready,
        "error": init_error,
        "dataset": REPO_ID,
        "files": stats["files"],
        "rows": stats["total_rows"],
        "queries_served": stats["queries"],
        "uptime_s": uptime,
    }
