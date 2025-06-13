import asyncpg # type: ignore
import pandas as pd
import asyncio
from fastapi import FastAPI, Depends, HTTPException, status # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel # type: ignore
from typing import List, Dict, Any, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv # type: ignore
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Eduroam Analytics API",
    description="API for fetching and visualizing Eduroam log data",
    version="3.0.0"
)

# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Add production domains here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

# Pydantic models
class TimeRange(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class AnalyticsResponse(BaseModel):
    labels: List[str]
    datasets: List[Dict[str, Any]]

# Globals
pool = None
executor = ThreadPoolExecutor(max_workers=10)

@app.on_event("startup")
async def startup():
    global pool
    pool = await asyncpg.create_pool(min_size=5, max_size=20, **DB_CONFIG) # type: ignore

@app.on_event("shutdown")
async def shutdown():
    if pool:
        await pool.close()

async def fetch_records(query: str, params: Optional[Sequence[Any]] = None) -> List[asyncpg.Record]:
    """Fetch records asynchronously from DB"""
    async with pool.acquire() as conn: # type: ignore
        return await conn.fetch(query, *(params or []))

async def run_in_thread(fn, *args, **kwargs):
    """Run blocking function in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: fn(*args, **kwargs))

@app.get("/top-users", response_model=AnalyticsResponse)
async def get_top_users(
    source_host: str,
    limit: int = 30,
    time_range: TimeRange = Depends()
):
    """Get top users by login count"""
    try:
        query = """
            SELECT username, log_timestamp 
            FROM eduroam_logs
            WHERE source_host = $1
            LIMIT $2
        """
        params = [source_host, limit]
        if time_range.start_date and time_range.end_date:
            query += " AND log_timestamp BETWEEN $2 AND $3"
            params.extend([time_range.start_date, time_range.end_date])

        records = await fetch_records(query, params)
        records_dict = [dict(r) for r in records]

        def process_top_users(records: List[Dict[str, Any]]) -> AnalyticsResponse:
            df = pd.DataFrame(records)
            if df.empty:
                return AnalyticsResponse(labels=[], datasets=[])
            top_users = df['username'].value_counts().head(limit)
            return AnalyticsResponse(
                labels=top_users.index.tolist(),
                datasets=[{
                    "label": "Login Count",
                    "data": top_users.values.tolist(),
                    "backgroundColor": "#4f46e5"
                }]
            )

        return await run_in_thread(process_top_users, records_dict)

    except Exception as e:
        print(f"Error in /top-users: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch top users")

@app.get("/logins-over-time", response_model=AnalyticsResponse)
async def get_logins_over_time(
    source_host: str,
    interval: str = "day",
    time_range: TimeRange = Depends()
):
    """Get login trends aggregated by interval"""
    try:
        freq_map = {
            "day": "D",
            "week": "W",
            "month": "M",
            "year": "Y"
        }
        if interval not in freq_map:
            raise HTTPException(
                status_code=400,
                detail="Invalid interval. Use day, week, month, or year"
            )

        query = """
            SELECT log_timestamp 
            FROM eduroam_logs
            WHERE source_host = $1
        """
        params = [source_host]
        if time_range.start_date and time_range.end_date:
            query += " AND log_timestamp BETWEEN $2 AND $3"
            params.extend([time_range.start_date, time_range.end_date])

        records = await fetch_records(query, params)
        records_dict = [dict(r) for r in records]

        def process_logins(records: List[Dict[str, Any]]) -> AnalyticsResponse:
            df = pd.DataFrame(records)
            if df.empty:
                return AnalyticsResponse(labels=[], datasets=[])

            df['log_timestamp'] = pd.to_datetime(df['log_timestamp'])
            df.set_index('log_timestamp', inplace=True)
            login_counts = df.resample(freq_map[interval]).size()

            # Format x-axis labels
            if interval == "day":
                labels = login_counts.index.strftime("%Y-%m-%d").tolist() # type: ignore
            elif interval == "week":
                labels = login_counts.index.strftime("Week %U, %Y").tolist() # type: ignore
            elif interval == "month":
                labels = login_counts.index.strftime("%b %Y").tolist() # type: ignore
            else:
                labels = login_counts.index.strftime("%Y").tolist() # type: ignore

            return AnalyticsResponse(
                labels=labels,
                datasets=[{
                    "label": "Logins",
                    "data": login_counts.values.tolist(),
                    "borderColor": "#4f46e5",
                    "fill": False
                }]
            )

        return await run_in_thread(process_logins, records_dict)

    except Exception as e:
        print(f"Error in /logins-over-time: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch login trends")

@app.get("/health")
async def health_check():
    try:
        async with pool.acquire() as conn: # type: ignore
            await conn.execute("SELECT 1")
            return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "degraded", "database": "disconnected", "error": str(e)}, 503

if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000)
