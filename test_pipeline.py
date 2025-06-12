#!/usr/bin/env python3
"""
Test script for the Tweet Processing Pipeline
This script demonstrates the pipeline functionality with a smaller dataset
"""

import asyncio
import time
from datetime import datetime, timezone

from loguru import logger

from dummy_data_generator import DummyTweetGenerator
from models import Tweet
from tweet_pipeline import TweetProcessingPipeline


async def test_basic_functionality():
    """Test basic pipeline functionality"""
    logger.info("=== Testing Basic Pipeline Functionality ===")

    # Initialize pipeline
    pipeline = TweetProcessingPipeline()

    # Generate small test dataset
    generator = DummyTweetGenerator()
    test_tweets = generator.generate_dummy_tweets(100)

    logger.info(f"Generated {len(test_tweets)} test tweets")

    # Process tweets in small batches
    batch_size = 10
    total_processed = 0

    for i in range(0, len(test_tweets), batch_size):
        batch = test_tweets[i : i + batch_size]
        results = pipeline.add_tweets_batch(batch)
        total_processed += results["processed"]

        logger.info(
            f"Batch {i//batch_size + 1}: Processed {results['processed']}, "
            f"Errors: {results['errors']}, Alerts: {results['alerts_sent']}"
        )

        # Small delay to simulate real-time
        await asyncio.sleep(1)

    # Get final statistics
    stats = pipeline.get_pipeline_stats()

    logger.info("=== Final Statistics ===")
    logger.info(f"Total tweets processed: {stats['tweets_processed']}")
    logger.info(f"Active clusters: {stats['active_clusters']}")
    logger.info(f"Clusters created: {stats['clusters_created']}")
    logger.info(f"Tweets clustered: {stats['tweets_clustered']}")
    logger.info(f"Alerts sent: {stats['alerts_sent']}")
    logger.info(f"Average cluster size: {stats['average_cluster_size']:.2f}")

    # Show some cluster details
    logger.info("=== Sample Clusters ===")
    for i, (cluster_id, cluster) in enumerate(
        list(pipeline.clustering_engine.clusters.items())[:5]
    ):
        summary = pipeline.clustering_engine.get_cluster_summary(
            cluster_id, pipeline.tweets
        )
        logger.info(
            f"Cluster {i+1}: {cluster.size} tweets, Topic: {cluster.topic_match}, "
            f"Similarity: {cluster.topic_similarity:.3f}"
        )
        logger.info(f"  Summary: {summary}")

    return pipeline


async def test_cluster_expiration():
    """Test cluster expiration functionality"""
    logger.info("=== Testing Cluster Expiration ===")

    pipeline = TweetProcessingPipeline()

    # Create some test tweets with old timestamps
    old_tweets = []
    for i in range(10):
        tweet = Tweet(
            tweet_id=f"old_tweet_{i}",
            text=f"This is an old tweet about technology #{i}",
            author_id=f"user_{i}",
            author_username=f"testuser{i}",
            created_at=datetime.now(timezone.utc).replace(day=1),  # Make it old
        )
        old_tweets.append(tweet)

    # Process old tweets
    pipeline.add_tweets_batch(old_tweets)

    logger.info(
        f"Created {len(pipeline.clustering_engine.clusters)} clusters with old tweets"
    )

    # Force expiration check
    expired = pipeline.clustering_engine.expire_old_clusters()

    logger.info(f"Expired {len(expired)} clusters")

    return pipeline


async def test_topic_matching():
    """Test topic matching functionality"""
    logger.info("=== Testing Topic Matching ===")

    pipeline = TweetProcessingPipeline()

    # Create tweets with specific topics
    topic_tweets = [
        Tweet(
            tweet_id="tech_1",
            text="Artificial intelligence is revolutionizing healthcare with machine learning algorithms",
            author_id="user_1",
            author_username="techexpert",
            created_at=datetime.now(timezone.utc),
        ),
        Tweet(
            tweet_id="tech_2",
            text="New breakthrough in deep learning models for natural language processing",
            author_id="user_2",
            author_username="airesearcher",
            created_at=datetime.now(timezone.utc),
        ),
        Tweet(
            tweet_id="market_1",
            text="Bitcoin price surges as institutional investors embrace cryptocurrency",
            author_id="user_3",
            author_username="cryptotrader",
            created_at=datetime.now(timezone.utc),
        ),
        Tweet(
            tweet_id="market_2",
            text="Stock market analysis shows tech stocks outperforming expectations",
            author_id="user_4",
            author_username="marketanalyst",
            created_at=datetime.now(timezone.utc),
        ),
    ]

    # Process topic-specific tweets
    results = pipeline.add_tweets_batch(topic_tweets)

    logger.info(f"Processed {results['processed']} topic-specific tweets")

    # Show topic matching results
    for cluster_id, cluster in pipeline.clustering_engine.clusters.items():
        if cluster.topic_match:
            logger.info(
                f"Cluster matched topic '{cluster.topic_match}' with similarity {cluster.topic_similarity:.3f}"
            )
            logger.info(f"  Cluster size: {cluster.size} tweets")

    return pipeline


async def run_all_tests():
    """Run all tests"""
    logger.info("Starting Tweet Processing Pipeline Tests")

    try:
        # Test basic functionality
        await test_basic_functionality()
        await asyncio.sleep(2)

        # Test topic matching
        await test_topic_matching()
        await asyncio.sleep(2)

        # Test cluster expiration
        await test_cluster_expiration()

        logger.info("All tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
