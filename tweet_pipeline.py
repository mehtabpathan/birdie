import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import redis
import schedule
from loguru import logger

from clustering_engine import IncrementalClusteringEngine
from config import config
from dummy_data_generator import DummyTweetGenerator
from models import Cluster, ClusterStatus, ProcessingStats, Tweet
from slack_alerter import SlackAlerter
from text_processor import TextProcessor


class TweetProcessingPipeline:
    def __init__(self):
        # Initialize components
        self.text_processor = TextProcessor()
        self.clustering_engine = IncrementalClusteringEngine()
        self.slack_alerter = SlackAlerter()

        # Initialize Redis for data persistence
        try:
            self.redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True,
            )
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
            self.redis_client = None

        # Data storage
        self.tweets: Dict[str, Tweet] = {}
        self.processing_stats = ProcessingStats()

        # Control flags
        self.is_running = False
        self.should_stop = False

        # For development - dummy data generator
        self.dummy_generator = DummyTweetGenerator()

        # Set tweets reference in clustering engine for quality maintenance
        self.clustering_engine.set_all_tweets_reference(self.tweets)

        logger.info("Tweet Processing Pipeline initialized")

    def _save_tweet_to_redis(self, tweet: Tweet):
        """Save tweet to Redis"""
        if self.redis_client:
            try:
                self.redis_client.hset(
                    "tweets", tweet.id, json.dumps(tweet.dict(), default=str)
                )
            except Exception as e:
                logger.error(f"Failed to save tweet to Redis: {e}")

    def _save_cluster_to_redis(self, cluster: Cluster):
        """Save cluster to Redis"""
        if self.redis_client:
            try:
                self.redis_client.hset(
                    "clusters", cluster.id, json.dumps(cluster.dict(), default=str)
                )
            except Exception as e:
                logger.error(f"Failed to save cluster to Redis: {e}")

    def _load_data_from_redis(self):
        """Load existing data from Redis"""
        if not self.redis_client:
            return

        try:
            # Load tweets
            tweet_data = self.redis_client.hgetall("tweets")
            for tweet_id, tweet_json in tweet_data.items():
                tweet_dict = json.loads(tweet_json)
                self.tweets[tweet_id] = Tweet(**tweet_dict)

            logger.info(f"Loaded {len(self.tweets)} tweets from Redis")

            # Load clusters (if needed for persistence)
            # This would require more complex logic to rebuild the clustering engine state

        except Exception as e:
            logger.error(f"Failed to load data from Redis: {e}")

    def add_tweet(self, tweet: Tweet) -> bool:
        """Add a new tweet to the pipeline"""
        try:
            # Store tweet
            self.tweets[tweet.id] = tweet
            self._save_tweet_to_redis(tweet)

            # Update clustering engine's tweets reference
            self.clustering_engine.set_all_tweets_reference(self.tweets)

            # Process through clustering engine
            cluster_id = self.clustering_engine.process_tweet(tweet)

            if cluster_id:
                # Save updated cluster
                cluster = self.clustering_engine.clusters[cluster_id]
                self._save_cluster_to_redis(cluster)

                # Check if cluster needs alerting
                if self._should_alert_cluster(cluster):
                    self._send_cluster_alert(cluster)

            # Update stats
            self.processing_stats.tweets_processed += 1
            self.processing_stats.last_processing_time = datetime.now(timezone.utc)

            return True

        except Exception as e:
            logger.error(f"Error processing tweet {tweet.id}: {e}")
            self.processing_stats.processing_errors += 1
            return False

    def add_tweets_batch(self, tweets: List[Tweet]) -> Dict[str, int]:
        """Add a batch of tweets"""
        results = {"processed": 0, "errors": 0, "new_clusters": 0, "alerts_sent": 0}

        logger.info(f"Processing batch of {len(tweets)} tweets")

        for tweet in tweets:
            if self.add_tweet(tweet):
                results["processed"] += 1
            else:
                results["errors"] += 1

        # Check for new clusters that need alerting
        alert_clusters = self.clustering_engine.get_clusters_for_alerting()
        for cluster in alert_clusters:
            if self._send_cluster_alert(cluster):
                results["alerts_sent"] += 1

        logger.info(f"Batch processing complete: {results}")
        return results

    def _should_alert_cluster(self, cluster: Cluster) -> bool:
        """Check if cluster meets alerting criteria"""
        return (
            cluster.status == ClusterStatus.ACTIVE
            and not cluster.alert_sent
            and cluster.size >= config.MIN_CLUSTER_SIZE
            and cluster.topic_match is not None
            and cluster.topic_similarity >= 0.6
        )

    def _send_cluster_alert(self, cluster: Cluster) -> bool:
        """Send alert for cluster"""
        try:
            success = self.slack_alerter.send_cluster_alert(cluster, self.tweets)

            if success:
                cluster.alert_sent = True
                cluster.status = ClusterStatus.ALERTED
                self._save_cluster_to_redis(cluster)
                self.processing_stats.alerts_sent += 1
                logger.info(f"Alert sent for cluster {cluster.id}")

            return success

        except Exception as e:
            logger.error(f"Error sending alert for cluster {cluster.id}: {e}")
            return False

    def expire_old_clusters(self):
        """Expire old clusters and clean up data"""
        try:
            expired_cluster_ids = self.clustering_engine.expire_old_clusters()

            if expired_cluster_ids:
                logger.info(f"Expired {len(expired_cluster_ids)} clusters")

                # Remove expired tweets from storage
                tweets_to_remove = []
                for cluster_id in expired_cluster_ids:
                    if cluster_id in self.clustering_engine.clusters:
                        cluster = self.clustering_engine.clusters[cluster_id]
                        tweets_to_remove.extend(cluster.tweet_ids)

                # Remove tweets
                for tweet_id in tweets_to_remove:
                    if tweet_id in self.tweets:
                        del self.tweets[tweet_id]
                        if self.redis_client:
                            self.redis_client.hdel("tweets", tweet_id)

                logger.info(f"Removed {len(tweets_to_remove)} expired tweets")

                # Update stats
                self.processing_stats.clusters_expired += len(expired_cluster_ids)

                # Rebuild index if many clusters expired
                if len(expired_cluster_ids) > 10:
                    self.clustering_engine.rebuild_index()

        except Exception as e:
            logger.error(f"Error expiring clusters: {e}")
            self.slack_alerter.send_error_alert(str(e), "expire_old_clusters")

    def get_pipeline_stats(self) -> Dict:
        """Get comprehensive pipeline statistics"""
        clustering_stats = self.clustering_engine.get_statistics()

        return {
            **self.processing_stats.dict(),
            **clustering_stats,
            "total_tweets_stored": len(self.tweets),
            "pipeline_uptime": time.time() - getattr(self, "start_time", time.time()),
        }

    def send_summary_report(self):
        """Send periodic summary report"""
        try:
            stats = self.get_pipeline_stats()
            self.slack_alerter.send_summary_alert(stats)
            logger.info("Summary report sent")
        except Exception as e:
            logger.error(f"Error sending summary report: {e}")

    def start_scheduled_tasks(self):
        """Start scheduled maintenance tasks"""
        # Schedule cluster expiration check every hour
        schedule.every().hour.do(self.expire_old_clusters)

        # Schedule summary report every 6 hours
        schedule.every(6).hours.do(self.send_summary_report)

        # Schedule index rebuild every day
        schedule.every().day.do(self.clustering_engine.rebuild_index)

        logger.info("Scheduled tasks configured")

    async def run_scheduled_tasks(self):
        """Run scheduled tasks in background"""
        while not self.should_stop:
            schedule.run_pending()
            await asyncio.sleep(60)  # Check every minute

    async def simulate_real_time_processing(self, tweet_rate: int = 10):
        """Simulate real-time tweet processing using dummy data"""
        logger.info(f"Starting real-time simulation with {tweet_rate} tweets per batch")

        # Load dummy tweets if not already loaded
        try:
            dummy_tweets = self.dummy_generator.load_tweets_from_json()
        except FileNotFoundError:
            logger.info("Generating dummy tweets...")
            dummy_tweets = self.dummy_generator.generate_dummy_tweets(50000)
            self.dummy_generator.save_tweets_to_json(dummy_tweets)

        # Process tweets in batches
        batch_size = tweet_rate
        total_tweets = len(dummy_tweets)
        processed = 0

        self.start_time = time.time()
        self.is_running = True

        # Start scheduled tasks
        self.start_scheduled_tasks()

        try:
            while processed < total_tweets and not self.should_stop:
                # Get next batch
                batch = dummy_tweets[processed : processed + batch_size]

                # Process batch
                results = self.add_tweets_batch(batch)
                processed += len(batch)

                # Log progress
                if processed % (batch_size * 10) == 0:
                    stats = self.get_pipeline_stats()
                    logger.info(
                        f"Processed {processed}/{total_tweets} tweets. "
                        f"Active clusters: {stats['active_clusters']}, "
                        f"Alerts sent: {stats['alerts_sent']}"
                    )

                # Wait before next batch (simulate real-time)
                await asyncio.sleep(config.PROCESSING_INTERVAL)

                # Run scheduled tasks
                schedule.run_pending()

        except KeyboardInterrupt:
            logger.info("Stopping pipeline...")
            self.should_stop = True

        finally:
            self.is_running = False
            logger.info("Pipeline stopped")

    async def start_pipeline(self, mode: str = "simulation"):
        """Start the pipeline"""
        logger.info(f"Starting pipeline in {mode} mode")

        # Load existing data
        self._load_data_from_redis()

        # Send test alert
        self.slack_alerter.test_alert()

        if mode == "simulation":
            await self.simulate_real_time_processing()
        else:
            # In production, this would connect to Twitter API or message queue
            logger.info("Production mode not implemented yet")

    def stop_pipeline(self):
        """Stop the pipeline gracefully"""
        logger.info("Stopping pipeline...")
        self.should_stop = True

        # Send final summary
        self.send_summary_report()

        logger.info("Pipeline stopped")


# Main execution
async def main():
    """Main function to run the pipeline"""
    pipeline = TweetProcessingPipeline()

    try:
        await pipeline.start_pipeline("simulation")
    except KeyboardInterrupt:
        pipeline.stop_pipeline()
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        pipeline.slack_alerter.send_error_alert(str(e), "main pipeline")


if __name__ == "__main__":
    asyncio.run(main())
