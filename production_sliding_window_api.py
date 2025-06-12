#!/usr/bin/env python3
"""
Production FastAPI server for Sliding Window Tweet Clustering
Features:
- Sliding window clustering with natural expiry
- Semantic alert deduplication with persistence
- S3 state backup and recovery
- Comprehensive monitoring and health checks
- RESTful API endpoints
"""

import asyncio
import json
import logging
import pickle
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

import boto3
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from config import config
from models import Cluster, Tweet
from sliding_window_clustering_engine import SlidingWindowClusteringEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Pydantic models for API
class TweetInput(BaseModel):
    id: str
    text: str
    author: str
    created_at: Optional[str] = None


class BatchTweetInput(BaseModel):
    tweets: List[TweetInput]


class ClusterResponse(BaseModel):
    cluster_id: str
    size: int
    topic_match: Optional[str]
    topic_similarity: float
    status: str
    created_at: str
    last_tweet_added: str
    sample_tweets: List[str]


class AlertResponse(BaseModel):
    alert_id: str
    cluster_id: str
    topic: str
    keywords: List[str]
    tweet_count: int
    timestamp: str
    reason: str


class SystemStats(BaseModel):
    tweets_processed: int
    active_clusters: int
    active_windows: int
    clusters_created: int
    clusters_expired: int
    windows_created: int
    windows_expired: int
    alerts_generated: int
    alerts_deduplicated: int
    total_alerts: int
    recent_alerts_24h: int
    uptime_seconds: float


class HealthCheck(BaseModel):
    status: str
    timestamp: str
    engine_status: str
    s3_connectivity: bool
    alert_deduplicator_status: str
    memory_usage_mb: float


class S3StateManager:
    """Manages state persistence to S3 for the sliding window engine"""

    def __init__(self):
        self.s3_client = boto3.client("s3")
        self.bucket_name = config.S3_BUCKET_NAME
        self.state_key = "sliding_window_clustering_state.pkl"
        self.alert_history_key = "alert_history_backup.pkl"

    async def save_engine_state(self, engine: SlidingWindowClusteringEngine) -> bool:
        """Save engine state to S3"""
        try:
            # Prepare state data
            state_data = {
                "active_clusters": {
                    cluster_id: {
                        "cluster": cluster,
                        "embedding": engine.cluster_embeddings.get(cluster_id),
                        "tweets": engine.cluster_tweets.get(cluster_id, []),
                    }
                    for cluster_id, cluster in engine.active_clusters.items()
                },
                "windows": engine.windows,
                "window_start_times": engine.window_start_times,
                "current_window_id": engine.current_window_id,
                "stats": engine.stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Serialize and upload
            serialized_data = pickle.dumps(state_data)

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.state_key,
                Body=serialized_data,
                ContentType="application/octet-stream",
            )

            logger.info(f"Engine state saved to S3: {len(serialized_data)} bytes")
            return True

        except Exception as e:
            logger.error(f"Failed to save engine state to S3: {e}")
            return False

    async def load_engine_state(self, engine: SlidingWindowClusteringEngine) -> bool:
        """Load engine state from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=self.state_key
            )

            state_data = pickle.loads(response["Body"].read())

            # Restore engine state
            engine.active_clusters = {}
            engine.cluster_embeddings = {}
            engine.cluster_tweets = {}

            for cluster_id, cluster_data in state_data["active_clusters"].items():
                engine.active_clusters[cluster_id] = cluster_data["cluster"]
                if cluster_data["embedding"] is not None:
                    engine.cluster_embeddings[cluster_id] = cluster_data["embedding"]
                engine.cluster_tweets[cluster_id] = cluster_data["tweets"]

            engine.windows = state_data["windows"]
            engine.window_start_times = state_data["window_start_times"]
            engine.current_window_id = state_data["current_window_id"]
            engine.stats = state_data["stats"]

            # Rebuild FAISS index
            engine._rebuild_faiss_index()

            logger.info(
                f"Engine state loaded from S3 (saved: {state_data['timestamp']})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load engine state from S3: {e}")
            return False

    async def backup_alert_history(self, alert_deduplicator) -> bool:
        """Backup alert history to S3"""
        try:
            backup_data = {
                "alert_history": alert_deduplicator.alert_history,
                "alert_embeddings": alert_deduplicator.alert_embeddings,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            serialized_data = pickle.dumps(backup_data)

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.alert_history_key,
                Body=serialized_data,
                ContentType="application/octet-stream",
            )

            logger.info(
                f"Alert history backed up to S3: {len(backup_data['alert_history'])} alerts"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to backup alert history to S3: {e}")
            return False


# Global variables
clustering_engine: Optional[SlidingWindowClusteringEngine] = None
s3_manager: Optional[S3StateManager] = None
app_start_time: datetime = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global clustering_engine, s3_manager

    logger.info("🚀 Starting Sliding Window Tweet Clustering API")

    # Initialize S3 state manager
    s3_manager = S3StateManager()

    # Initialize clustering engine
    clustering_engine = SlidingWindowClusteringEngine(
        window_size_hours=config.CLUSTER_EXPIRY_HOURS,
        window_overlap_hours=config.CLUSTER_EXPIRY_HOURS // 4,
    )

    # Try to load previous state
    if await s3_manager.load_engine_state(clustering_engine):
        logger.info("✅ Previous engine state loaded successfully")
    else:
        logger.info("🆕 Starting with fresh engine state")

    # Start background tasks
    asyncio.create_task(periodic_backup())
    asyncio.create_task(periodic_maintenance())

    logger.info("✅ Sliding Window Clustering API ready")

    yield

    # Cleanup on shutdown
    logger.info("🛑 Shutting down Sliding Window Clustering API")

    if clustering_engine and s3_manager:
        await s3_manager.save_engine_state(clustering_engine)
        await s3_manager.backup_alert_history(clustering_engine.alert_deduplicator)
        clustering_engine.cleanup()

    logger.info("✅ Shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="Sliding Window Tweet Clustering API",
    description="Production API for sliding window tweet clustering with semantic alert deduplication",
    version="1.0.0",
    lifespan=lifespan,
)


async def periodic_backup():
    """Periodic backup of engine state to S3"""
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour

            if clustering_engine and s3_manager:
                await s3_manager.save_engine_state(clustering_engine)
                await s3_manager.backup_alert_history(
                    clustering_engine.alert_deduplicator
                )
                logger.info("🔄 Periodic backup completed")

        except Exception as e:
            logger.error(f"Error in periodic backup: {e}")


async def periodic_maintenance():
    """Periodic maintenance tasks"""
    while True:
        try:
            await asyncio.sleep(1800)  # Every 30 minutes

            if clustering_engine:
                # Force window expiry check
                clustering_engine._expire_old_windows()

                # Log statistics
                stats = clustering_engine.get_statistics()
                logger.info(
                    f"📊 Maintenance stats: {stats['active_clusters']} clusters, "
                    f"{stats['active_windows']} windows, "
                    f"{stats['alerts_generated']} alerts generated"
                )

        except Exception as e:
            logger.error(f"Error in periodic maintenance: {e}")


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    import psutil

    # Check S3 connectivity
    s3_ok = False
    try:
        if s3_manager:
            s3_manager.s3_client.head_bucket(Bucket=s3_manager.bucket_name)
            s3_ok = True
    except Exception:
        pass

    # Get memory usage
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024

    return HealthCheck(
        status="healthy" if clustering_engine else "unhealthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        engine_status="active" if clustering_engine else "inactive",
        s3_connectivity=s3_ok,
        alert_deduplicator_status=(
            "active"
            if clustering_engine and clustering_engine.alert_deduplicator
            else "inactive"
        ),
        memory_usage_mb=memory_mb,
    )


@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get comprehensive system statistics"""
    if not clustering_engine:
        raise HTTPException(status_code=503, detail="Clustering engine not available")

    stats = clustering_engine.get_statistics()
    uptime = (datetime.now(timezone.utc) - app_start_time).total_seconds()

    return SystemStats(
        tweets_processed=stats.get("tweets_processed", 0),
        active_clusters=stats.get("active_clusters", 0),
        active_windows=stats.get("active_windows", 0),
        clusters_created=stats.get("clusters_created", 0),
        clusters_expired=stats.get("clusters_expired", 0),
        windows_created=stats.get("windows_created", 0),
        windows_expired=stats.get("windows_expired", 0),
        alerts_generated=stats.get("alerts_generated", 0),
        alerts_deduplicated=stats.get("alerts_deduplicated", 0),
        total_alerts=stats.get("total_alerts", 0),
        recent_alerts_24h=stats.get("recent_alerts_24h", 0),
        uptime_seconds=uptime,
    )


@app.post("/tweets")
async def add_tweet(tweet_input: TweetInput, background_tasks: BackgroundTasks):
    """Add a single tweet for clustering"""
    if not clustering_engine:
        raise HTTPException(status_code=503, detail="Clustering engine not available")

    # Convert to Tweet object
    tweet = Tweet(
        id=tweet_input.id,
        text=tweet_input.text,
        author=tweet_input.author,
        created_at=(
            datetime.fromisoformat(tweet_input.created_at)
            if tweet_input.created_at
            else datetime.now(timezone.utc)
        ),
        cleaned_text=tweet_input.text.lower(),  # Simple cleaning
    )

    # Process tweet
    cluster_id = clustering_engine.process_tweet(tweet)

    # Schedule background backup if needed
    if clustering_engine.stats["tweets_processed"] % 100 == 0:
        background_tasks.add_task(s3_manager.save_engine_state, clustering_engine)

    return {
        "tweet_id": tweet.id,
        "cluster_id": cluster_id,
        "status": "clustered" if cluster_id else "skipped",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/tweets/batch")
async def add_tweets_batch(
    batch_input: BatchTweetInput, background_tasks: BackgroundTasks
):
    """Add multiple tweets for clustering"""
    if not clustering_engine:
        raise HTTPException(status_code=503, detail="Clustering engine not available")

    results = []

    for tweet_input in batch_input.tweets:
        tweet = Tweet(
            id=tweet_input.id,
            text=tweet_input.text,
            author=tweet_input.author,
            created_at=(
                datetime.fromisoformat(tweet_input.created_at)
                if tweet_input.created_at
                else datetime.now(timezone.utc)
            ),
            cleaned_text=tweet_input.text.lower(),
        )

        cluster_id = clustering_engine.process_tweet(tweet)

        results.append(
            {
                "tweet_id": tweet.id,
                "cluster_id": cluster_id,
                "status": "clustered" if cluster_id else "skipped",
            }
        )

    # Schedule background backup
    background_tasks.add_task(s3_manager.save_engine_state, clustering_engine)

    return {
        "processed": len(results),
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/clusters", response_model=List[ClusterResponse])
async def get_active_clusters():
    """Get all active clusters"""
    if not clustering_engine:
        raise HTTPException(status_code=503, detail="Clustering engine not available")

    clusters = []

    for cluster_id, cluster in clustering_engine.active_clusters.items():
        # Get sample tweets
        cluster_tweets = clustering_engine.cluster_tweets.get(cluster_id, [])
        sample_tweets = [tweet.text for tweet in cluster_tweets[:3]]

        clusters.append(
            ClusterResponse(
                cluster_id=cluster.id,
                size=cluster.size,
                topic_match=cluster.topic_match,
                topic_similarity=cluster.topic_similarity,
                status=cluster.status.value,
                created_at=cluster.created_at.isoformat(),
                last_tweet_added=cluster.last_tweet_added.isoformat(),
                sample_tweets=sample_tweets,
            )
        )

    return clusters


@app.get("/alerts", response_model=List[AlertResponse])
async def get_pending_alerts():
    """Get clusters ready for alerting"""
    if not clustering_engine:
        raise HTTPException(status_code=503, detail="Clustering engine not available")

    alert_candidates = clustering_engine.get_clusters_for_alerting()
    alerts = []

    for cluster, cluster_tweets, reason in alert_candidates:
        # Extract keywords
        keywords = clustering_engine.alert_deduplicator._extract_keywords(
            cluster_tweets
        )

        alerts.append(
            AlertResponse(
                alert_id=f"alert_{cluster.id}_{int(datetime.now().timestamp())}",
                cluster_id=cluster.id,
                topic=cluster.topic_match or "general",
                keywords=keywords,
                tweet_count=cluster.size,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason=reason,
            )
        )

    return alerts


@app.post("/backup")
async def manual_backup():
    """Manually trigger state backup to S3"""
    if not clustering_engine or not s3_manager:
        raise HTTPException(status_code=503, detail="Services not available")

    engine_backup = await s3_manager.save_engine_state(clustering_engine)
    alert_backup = await s3_manager.backup_alert_history(
        clustering_engine.alert_deduplicator
    )

    return {
        "engine_backup": engine_backup,
        "alert_backup": alert_backup,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/restore")
async def manual_restore():
    """Manually restore state from S3"""
    if not clustering_engine or not s3_manager:
        raise HTTPException(status_code=503, detail="Services not available")

    success = await s3_manager.load_engine_state(clustering_engine)

    return {"success": success, "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/windows")
async def get_window_info():
    """Get information about active windows"""
    if not clustering_engine:
        raise HTTPException(status_code=503, detail="Clustering engine not available")

    windows_info = []

    for window_id, start_time in clustering_engine.window_start_times.items():
        window_tweets = clustering_engine.windows.get(window_id, [])

        windows_info.append(
            {
                "window_id": window_id,
                "start_time": start_time.isoformat(),
                "tweet_count": len(window_tweets),
                "age_hours": (datetime.now(timezone.utc) - start_time).total_seconds()
                / 3600,
            }
        )

    return {
        "active_windows": len(windows_info),
        "windows": windows_info,
        "window_size_hours": clustering_engine.window_size_hours,
        "window_overlap_hours": clustering_engine.window_overlap_hours,
    }


if __name__ == "__main__":
    uvicorn.run(
        "production_sliding_window_api:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for state consistency
        log_level="info",
        access_log=True,
    )
