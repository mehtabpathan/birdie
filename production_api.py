#!/usr/bin/env python3
"""
Production FastAPI server for BIRCH Tweet Clustering System
Features:
- RESTful API endpoints
- S3 state persistence
- Health checks and monitoring
- Configuration management
- Graceful shutdown with state saving
"""

import asyncio
import json
import os
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

import boto3
import uvicorn
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from cluster_visualizer import ClusterVisualizer
from clustering_engine import IncrementalClusteringEngine
from config import config
from models import Cluster, Tweet
from tweet_pipeline import TweetProcessingPipeline


# Pydantic models for API
class TweetInput(BaseModel):
    tweet_id: str
    text: str
    author_id: str
    author_username: str
    created_at: Optional[datetime] = None
    retweet_count: int = 0
    like_count: int = 0
    reply_count: int = 0
    quote_count: int = 0


class BatchTweetInput(BaseModel):
    tweets: List[TweetInput]


class ClusterResponse(BaseModel):
    cluster_id: str
    size: int
    topic_match: Optional[str]
    topic_similarity: float
    coherence: float
    created_at: datetime
    last_updated: datetime
    status: str


class SystemStats(BaseModel):
    tweets_processed: int
    active_clusters: int
    total_clusters: int
    tweets_clustered: int
    average_cluster_size: float
    system_uptime: float
    last_backup: Optional[datetime]
    s3_enabled: bool


class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]


# S3 State Manager
class S3StateManager:
    def __init__(self):
        self.s3_enabled = False
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.s3_prefix = os.getenv("S3_PREFIX", "birch-clustering")

        if self.bucket_name:
            try:
                self.s3_client = boto3.client("s3")
                # Test connection
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                self.s3_enabled = True
                logger.info(
                    f"S3 state persistence enabled: s3://{self.bucket_name}/{self.s3_prefix}"
                )
            except (ClientError, NoCredentialsError) as e:
                logger.warning(f"S3 not available: {e}. State will not be persisted.")
                self.s3_client = None
        else:
            logger.warning(
                "S3_BUCKET_NAME not configured. State will not be persisted."
            )
            self.s3_client = None

    async def save_state(self, pipeline: TweetProcessingPipeline) -> bool:
        """Save complete system state to S3"""
        if not self.s3_enabled:
            logger.warning("S3 not enabled, skipping state save")
            return False

        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            # Prepare state data
            state_data = {
                "timestamp": timestamp,
                "version": "1.0",
                "config": {
                    "embedding_model": config.EMBEDDING_MODEL,
                    "min_cluster_size": config.MIN_CLUSTER_SIZE,
                    "similarity_threshold": config.SIMILARITY_THRESHOLD,
                    "cluster_expiry_hours": config.CLUSTER_EXPIRY_HOURS,
                },
                "pipeline_stats": pipeline.get_pipeline_stats(),
                "clustering_stats": pipeline.clustering_engine.get_statistics(),
            }

            # Save clustering engine state
            clustering_state = {
                "clusters": {
                    k: v.dict() for k, v in pipeline.clustering_engine.clusters.items()
                },
                "cluster_id_to_birch_label": pipeline.clustering_engine.cluster_id_to_birch_label,
                "birch_label_to_cluster_id": pipeline.clustering_engine.birch_label_to_cluster_id,
                "stats": pipeline.clustering_engine.stats,
                "birch_fitted": pipeline.clustering_engine.birch_fitted,
            }

            # Save tweets (sample for large datasets)
            tweets_to_save = dict(list(pipeline.tweets.items())[:10000])  # Limit for S3
            tweets_state = {k: v.dict() for k, v in tweets_to_save.items()}

            # Save embeddings (compressed)
            embeddings_state = {
                "tweet_embeddings_count": len(
                    pipeline.clustering_engine.tweet_embeddings
                ),
                "cluster_embeddings_count": len(
                    pipeline.clustering_engine.cluster_embeddings
                ),
                # Note: Full embeddings would be too large for JSON, use pickle for binary data
            }

            # Upload main state
            state_key = f"{self.s3_prefix}/state/{timestamp}/main_state.json"
            await self._upload_json(state_key, state_data)

            # Upload clustering state
            clustering_key = f"{self.s3_prefix}/state/{timestamp}/clustering_state.json"
            await self._upload_json(clustering_key, clustering_state)

            # Upload tweets state
            tweets_key = f"{self.s3_prefix}/state/{timestamp}/tweets_state.json"
            await self._upload_json(tweets_key, tweets_state)

            # Upload embeddings (binary)
            embeddings_key = f"{self.s3_prefix}/state/{timestamp}/embeddings.pkl"
            embeddings_data = {
                "tweet_embeddings": pipeline.clustering_engine.tweet_embeddings,
                "cluster_embeddings": pipeline.clustering_engine.cluster_embeddings,
                "tweet_birch_labels": pipeline.clustering_engine.tweet_birch_labels,
            }
            await self._upload_pickle(embeddings_key, embeddings_data)

            # Update latest state pointer
            latest_key = f"{self.s3_prefix}/latest_state.json"
            latest_data = {
                "timestamp": timestamp,
                "state_path": f"{self.s3_prefix}/state/{timestamp}/",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await self._upload_json(latest_key, latest_data)

            logger.info(f"State saved to S3: {timestamp}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state to S3: {e}")
            return False

    async def load_latest_state(self, pipeline: TweetProcessingPipeline) -> bool:
        """Load the latest state from S3"""
        if not self.s3_enabled:
            logger.warning("S3 not enabled, skipping state load")
            return False

        try:
            # Get latest state info
            latest_key = f"{self.s3_prefix}/latest_state.json"
            latest_data = await self._download_json(latest_key)

            if not latest_data:
                logger.info("No previous state found in S3")
                return False

            state_path = latest_data["state_path"]
            logger.info(f"Loading state from: {latest_data['timestamp']}")

            # Load main state
            main_state = await self._download_json(f"{state_path}main_state.json")
            if not main_state:
                logger.error("Failed to load main state")
                return False

            # Load clustering state
            clustering_state = await self._download_json(
                f"{state_path}clustering_state.json"
            )
            if clustering_state:
                # Restore clusters
                pipeline.clustering_engine.clusters = {}
                for cluster_id, cluster_data in clustering_state["clusters"].items():
                    cluster = Cluster(**cluster_data)
                    pipeline.clustering_engine.clusters[cluster_id] = cluster

                # Restore mappings
                pipeline.clustering_engine.cluster_id_to_birch_label = (
                    clustering_state.get("cluster_id_to_birch_label", {})
                )
                pipeline.clustering_engine.birch_label_to_cluster_id = (
                    clustering_state.get("birch_label_to_cluster_id", {})
                )
                pipeline.clustering_engine.stats = clustering_state.get("stats", {})
                pipeline.clustering_engine.birch_fitted = clustering_state.get(
                    "birch_fitted", False
                )

            # Load tweets state
            tweets_state = await self._download_json(f"{state_path}tweets_state.json")
            if tweets_state:
                pipeline.tweets = {}
                for tweet_id, tweet_data in tweets_state.items():
                    tweet = Tweet(**tweet_data)
                    pipeline.tweets[tweet_id] = tweet

            # Load embeddings
            embeddings_data = await self._download_pickle(f"{state_path}embeddings.pkl")
            if embeddings_data:
                pipeline.clustering_engine.tweet_embeddings = embeddings_data.get(
                    "tweet_embeddings", {}
                )
                pipeline.clustering_engine.cluster_embeddings = embeddings_data.get(
                    "cluster_embeddings", {}
                )
                pipeline.clustering_engine.tweet_birch_labels = embeddings_data.get(
                    "tweet_birch_labels", {}
                )

            # Update references
            pipeline.clustering_engine.set_all_tweets_reference(pipeline.tweets)

            logger.info(f"State loaded successfully from {latest_data['timestamp']}")
            logger.info(
                f"Restored: {len(pipeline.tweets)} tweets, {len(pipeline.clustering_engine.clusters)} clusters"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load state from S3: {e}")
            return False

    async def _upload_json(self, key: str, data: dict):
        """Upload JSON data to S3"""
        json_data = json.dumps(data, default=str, indent=2)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json_data.encode("utf-8"),
            ContentType="application/json",
        )

    async def _upload_pickle(self, key: str, data):
        """Upload pickle data to S3"""
        pickle_data = pickle.dumps(data)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=pickle_data,
            ContentType="application/octet-stream",
        )

    async def _download_json(self, key: str) -> Optional[dict]:
        """Download JSON data from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            data = json.loads(response["Body"].read().decode("utf-8"))
            return data
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    async def _download_pickle(self, key: str):
        """Download pickle data from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            data = pickle.loads(response["Body"].read())
            return data
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise


# Global state
pipeline: Optional[TweetProcessingPipeline] = None
s3_manager: Optional[S3StateManager] = None
start_time = time.time()
last_backup_time: Optional[datetime] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global pipeline, s3_manager, last_backup_time

    logger.info("🚀 Starting BIRCH Clustering API...")

    # Initialize components
    pipeline = TweetProcessingPipeline()
    s3_manager = S3StateManager()

    # Load previous state if available
    if s3_manager.s3_enabled:
        logger.info("Loading previous state from S3...")
        await s3_manager.load_latest_state(pipeline)

    # Start background tasks
    asyncio.create_task(periodic_backup())
    asyncio.create_task(periodic_maintenance())

    logger.info("✅ API ready to serve requests")

    yield

    # Shutdown: Save state
    logger.info("💾 Saving state before shutdown...")
    if s3_manager and s3_manager.s3_enabled:
        await s3_manager.save_state(pipeline)
    logger.info("👋 Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="BIRCH Tweet Clustering API",
    description="Production API for real-time tweet clustering using BIRCH algorithm",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Background tasks
async def periodic_backup():
    """Periodic state backup to S3"""
    global last_backup_time

    while True:
        try:
            await asyncio.sleep(3600)  # Every hour

            if s3_manager and s3_manager.s3_enabled and pipeline:
                logger.info("🔄 Performing periodic backup...")
                success = await s3_manager.save_state(pipeline)
                if success:
                    last_backup_time = datetime.now(timezone.utc)
                    logger.info("✅ Periodic backup completed")
                else:
                    logger.error("❌ Periodic backup failed")

        except Exception as e:
            logger.error(f"Error in periodic backup: {e}")


async def periodic_maintenance():
    """Periodic system maintenance"""
    while True:
        try:
            await asyncio.sleep(1800)  # Every 30 minutes

            if pipeline:
                # Expire old clusters
                expired = pipeline.expire_old_clusters()
                if expired:
                    logger.info(f"🧹 Expired {len(expired)} old clusters")

                # Quality maintenance
                pipeline.clustering_engine._perform_quality_maintenance()

        except Exception as e:
            logger.error(f"Error in periodic maintenance: {e}")


# API Endpoints


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    components = {
        "pipeline": "healthy" if pipeline else "not_initialized",
        "clustering_engine": (
            "healthy" if pipeline and pipeline.clustering_engine else "not_initialized"
        ),
        "s3_persistence": (
            "enabled" if s3_manager and s3_manager.s3_enabled else "disabled"
        ),
        "redis": "connected" if pipeline and pipeline.redis_client else "disconnected",
    }

    overall_status = (
        "healthy"
        if all(
            status in ["healthy", "enabled", "connected", "disabled"]
            for status in components.values()
        )
        else "unhealthy"
    )

    return HealthCheck(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version="1.0.0",
        components=components,
    )


@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get comprehensive system statistics"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    stats = pipeline.get_pipeline_stats()
    clustering_stats = pipeline.clustering_engine.get_statistics()

    return SystemStats(
        tweets_processed=stats["tweets_processed"],
        active_clusters=clustering_stats["active_clusters"],
        total_clusters=clustering_stats["total_clusters"],
        tweets_clustered=stats["tweets_clustered"],
        average_cluster_size=clustering_stats["average_cluster_size"],
        system_uptime=time.time() - start_time,
        last_backup=last_backup_time,
        s3_enabled=s3_manager.s3_enabled if s3_manager else False,
    )


@app.post("/tweets")
async def add_tweet(tweet_input: TweetInput, background_tasks: BackgroundTasks):
    """Add a single tweet for processing"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Convert input to Tweet model
        tweet = Tweet(
            tweet_id=tweet_input.tweet_id,
            text=tweet_input.text,
            author_id=tweet_input.author_id,
            author_username=tweet_input.author_username,
            created_at=tweet_input.created_at or datetime.now(timezone.utc),
            retweet_count=tweet_input.retweet_count,
            like_count=tweet_input.like_count,
            reply_count=tweet_input.reply_count,
            quote_count=tweet_input.quote_count,
        )

        # Process tweet
        success = pipeline.add_tweet(tweet)

        if success:
            cluster_id = tweet.cluster_id
            response = {
                "status": "success",
                "tweet_id": tweet.id,
                "cluster_id": cluster_id,
                "message": "Tweet processed successfully",
            }

            # Trigger backup if significant changes
            if pipeline.clustering_engine.stats["tweets_processed"] % 1000 == 0:
                background_tasks.add_task(trigger_backup)

            return response
        else:
            raise HTTPException(status_code=400, detail="Failed to process tweet")

    except Exception as e:
        logger.error(f"Error processing tweet: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tweets/batch")
async def add_tweets_batch(
    batch_input: BatchTweetInput, background_tasks: BackgroundTasks
):
    """Add multiple tweets for batch processing"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Convert inputs to Tweet models
        tweets = []
        for tweet_input in batch_input.tweets:
            tweet = Tweet(
                tweet_id=tweet_input.tweet_id,
                text=tweet_input.text,
                author_id=tweet_input.author_id,
                author_username=tweet_input.author_username,
                created_at=tweet_input.created_at or datetime.now(timezone.utc),
                retweet_count=tweet_input.retweet_count,
                like_count=tweet_input.like_count,
                reply_count=tweet_input.reply_count,
                quote_count=tweet_input.quote_count,
            )
            tweets.append(tweet)

        # Process batch
        results = pipeline.add_tweets_batch(tweets)

        # Trigger backup for large batches
        if len(tweets) > 100:
            background_tasks.add_task(trigger_backup)

        return {
            "status": "success",
            "processed": len(tweets),
            "results": results,
            "message": f"Processed {len(tweets)} tweets successfully",
        }

    except Exception as e:
        logger.error(f"Error processing tweet batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clusters")
async def get_clusters(
    limit: int = 50,
    status_filter: Optional[str] = None,
    topic_filter: Optional[str] = None,
):
    """Get clusters with optional filtering"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        clusters = []
        for cluster_id, cluster in pipeline.clustering_engine.clusters.items():
            # Apply filters
            if status_filter and cluster.status.value != status_filter:
                continue
            if topic_filter and cluster.topic_match != topic_filter:
                continue

            # Calculate coherence
            coherence = pipeline.clustering_engine._calculate_cluster_coherence(
                cluster_id
            )

            cluster_response = ClusterResponse(
                cluster_id=cluster_id,
                size=cluster.size,
                topic_match=cluster.topic_match,
                topic_similarity=cluster.topic_similarity or 0.0,
                coherence=coherence,
                created_at=cluster.created_at,
                last_updated=cluster.last_updated,
                status=cluster.status.value,
            )
            clusters.append(cluster_response)

        # Sort by size (largest first) and limit
        clusters.sort(key=lambda x: x.size, reverse=True)
        return clusters[:limit]

    except Exception as e:
        logger.error(f"Error getting clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clusters/{cluster_id}")
async def get_cluster_details(cluster_id: str):
    """Get detailed information about a specific cluster"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if cluster_id not in pipeline.clustering_engine.clusters:
        raise HTTPException(status_code=404, detail="Cluster not found")

    try:
        cluster = pipeline.clustering_engine.clusters[cluster_id]
        coherence = pipeline.clustering_engine._calculate_cluster_coherence(cluster_id)

        # Get tweets in cluster
        cluster_tweets = []
        for tweet_id in cluster.tweet_ids[:10]:  # Limit to first 10 tweets
            if tweet_id in pipeline.tweets:
                tweet = pipeline.tweets[tweet_id]
                cluster_tweets.append(
                    {
                        "tweet_id": tweet.tweet_id,
                        "text": tweet.text,
                        "author_username": tweet.author_username,
                        "created_at": tweet.created_at,
                        "engagement": {
                            "retweets": tweet.retweet_count,
                            "likes": tweet.like_count,
                            "replies": tweet.reply_count,
                        },
                    }
                )

        # Get BIRCH label
        birch_label = pipeline.clustering_engine.cluster_id_to_birch_label.get(
            cluster_id
        )

        return {
            "cluster_id": cluster_id,
            "size": cluster.size,
            "status": cluster.status.value,
            "topic_match": cluster.topic_match,
            "topic_similarity": cluster.topic_similarity or 0.0,
            "coherence": coherence,
            "created_at": cluster.created_at,
            "last_updated": cluster.last_updated,
            "last_tweet_added": cluster.last_tweet_added,
            "birch_label": birch_label,
            "alert_sent": cluster.alert_sent,
            "summary": pipeline.clustering_engine.get_cluster_summary(
                cluster_id, pipeline.tweets
            ),
            "sample_tweets": cluster_tweets,
            "centroid": (
                cluster.centroid[:5] if cluster.centroid else None
            ),  # First 5 dimensions
        }

    except Exception as e:
        logger.error(f"Error getting cluster details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backup")
async def trigger_backup():
    """Manually trigger a state backup to S3"""
    if not s3_manager or not s3_manager.s3_enabled:
        raise HTTPException(status_code=503, detail="S3 backup not configured")

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        success = await s3_manager.save_state(pipeline)
        if success:
            global last_backup_time
            last_backup_time = datetime.now(timezone.utc)
            return {
                "status": "success",
                "message": "Backup completed successfully",
                "timestamp": last_backup_time,
            }
        else:
            raise HTTPException(status_code=500, detail="Backup failed")

    except Exception as e:
        logger.error(f"Error during manual backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/restore")
async def restore_from_backup():
    """Restore system state from the latest S3 backup"""
    if not s3_manager or not s3_manager.s3_enabled:
        raise HTTPException(status_code=503, detail="S3 backup not configured")

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        success = await s3_manager.load_latest_state(pipeline)
        if success:
            return {"status": "success", "message": "State restored successfully"}
        else:
            return {"status": "warning", "message": "No backup found or restore failed"}

    except Exception as e:
        logger.error(f"Error during restore: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualizations/dashboard")
async def generate_dashboard():
    """Generate and return clustering dashboard"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        visualizer = ClusterVisualizer(pipeline.clustering_engine, pipeline.tweets)

        # Generate dashboard
        dashboard_path = "temp_dashboard.html"
        visualizer.create_comprehensive_dashboard(dashboard_path)

        # Read and return HTML content
        with open(dashboard_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Clean up temp file
        os.remove(dashboard_path)

        return JSONResponse(
            content={"html": html_content}, media_type="application/json"
        )

    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clusters/expired")
async def cleanup_expired_clusters():
    """Manually trigger cleanup of expired clusters"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        expired_ids = pipeline.expire_old_clusters()
        return {
            "status": "success",
            "expired_clusters": len(expired_ids),
            "cluster_ids": expired_ids,
            "message": f"Cleaned up {len(expired_ids)} expired clusters",
        }

    except Exception as e:
        logger.error(f"Error cleaning up clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Production server configuration
    uvicorn.run(
        "production_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=1,  # Single worker for state consistency
        log_level="info",
        access_log=True,
        reload=False,  # Disable in production
    )
