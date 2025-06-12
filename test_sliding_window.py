#!/usr/bin/env python3
"""
Test script for the Sliding Window Clustering Engine
Demonstrates natural cluster expiry and semantic alert deduplication
"""

import time
from datetime import datetime, timedelta, timezone
from typing import List

from models import Tweet
from sliding_window_clustering_engine import SlidingWindowClusteringEngine


def create_sample_tweets() -> List[Tweet]:
    """Create sample tweets for testing"""
    sample_data = [
        # Climate change cluster
        ("Climate change is affecting global weather patterns", "user1"),
        ("Global warming causes extreme weather events", "user2"),
        ("Rising temperatures impact ecosystems worldwide", "user3"),
        ("Climate crisis requires immediate action", "user4"),
        ("Greenhouse gases contribute to climate change", "user5"),
        # Technology cluster
        ("Artificial intelligence is transforming industries", "user6"),
        ("Machine learning algorithms improve daily", "user7"),
        ("AI technology advances rapidly", "user8"),
        ("Deep learning models show great potential", "user9"),
        ("Neural networks revolutionize computing", "user10"),
        # Sports cluster
        ("Football season starts with exciting matches", "user11"),
        ("Basketball playoffs are intense this year", "user12"),
        ("Soccer world cup preparations underway", "user13"),
        ("Tennis tournament features top players", "user14"),
        ("Olympic games showcase athletic excellence", "user15"),
        # Similar climate tweets (for deduplication testing)
        ("Climate change impacts are becoming severe", "user16"),
        ("Global warming effects are visible everywhere", "user17"),
        ("Environmental crisis needs urgent attention", "user18"),
        # Different topic
        ("Cooking recipes for healthy meals", "user19"),
        ("Nutrition tips for better health", "user20"),
    ]

    tweets = []
    base_time = datetime.now(timezone.utc)

    for i, (text, author) in enumerate(sample_data):
        tweet = Tweet(
            id=f"tweet_{i+1}",
            text=text,
            author=author,
            created_at=base_time + timedelta(minutes=i * 5),  # 5 minutes apart
            cleaned_text=text.lower(),  # Simple cleaning
        )
        tweets.append(tweet)

    return tweets


def test_sliding_window_clustering():
    """Test the sliding window clustering engine"""
    print("🚀 Testing Sliding Window Clustering Engine")
    print("=" * 60)

    # Initialize engine with short windows for testing
    engine = SlidingWindowClusteringEngine(
        window_size_hours=2,  # 2-hour windows
        window_overlap_hours=0.5,  # 30-minute overlap
    )

    # Create sample tweets
    tweets = create_sample_tweets()

    print(f"📝 Processing {len(tweets)} sample tweets...")
    print()

    # Process tweets in batches to simulate real-time processing
    batch_size = 5
    for i in range(0, len(tweets), batch_size):
        batch = tweets[i : i + batch_size]

        print(f"📦 Processing batch {i//batch_size + 1} ({len(batch)} tweets)")

        for tweet in batch:
            cluster_id = engine.process_tweet(tweet)
            if cluster_id:
                print(f"  ✅ Tweet '{tweet.text[:50]}...' → Cluster {cluster_id}")
            else:
                print(f"  ❌ Tweet '{tweet.text[:50]}...' → Skipped")

        # Get current statistics
        stats = engine.get_statistics()
        print(
            f"  📊 Stats: {stats['active_clusters']} clusters, "
            f"{stats['active_windows']} windows, "
            f"{stats['tweets_processed']} tweets processed"
        )
        print()

        # Small delay to simulate time passing
        time.sleep(0.1)

    print("🔍 Final Clustering Results")
    print("-" * 40)

    # Display final statistics
    final_stats = engine.get_statistics()
    print("📈 Final Statistics:")
    for key, value in final_stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    print()

    # Test alerting with deduplication
    print("🚨 Testing Alert Generation & Deduplication")
    print("-" * 50)

    alert_candidates = engine.get_clusters_for_alerting()

    if alert_candidates:
        print(f"Found {len(alert_candidates)} clusters ready for alerting:")
        for i, (cluster, cluster_tweets, reason) in enumerate(alert_candidates, 1):
            print(f"\n🔔 Alert {i}:")
            print(f"  Cluster ID: {cluster.id}")
            print(f"  Topic: {cluster.topic_match}")
            print(f"  Size: {cluster.size} tweets")
            print(f"  Topic Similarity: {cluster.topic_similarity:.3f}")
            print(f"  Reason: {reason}")
            print(f"  Sample tweets:")
            for tweet in cluster_tweets[:3]:  # Show first 3 tweets
                print(f"    - {tweet.text}")
    else:
        print("No clusters meet alerting criteria or all were deduplicated")

    # Test alert deduplication by processing similar tweets again
    print("\n🔄 Testing Alert Deduplication with Similar Content")
    print("-" * 55)

    # Create similar tweets to test deduplication
    similar_tweets = [
        Tweet(
            id="dup_tweet_1",
            text="Climate change is causing severe weather patterns globally",
            author="dup_user1",
            created_at=datetime.now(timezone.utc),
            cleaned_text="climate change is causing severe weather patterns globally",
        ),
        Tweet(
            id="dup_tweet_2",
            text="Global warming leads to extreme climate events",
            author="dup_user2",
            created_at=datetime.now(timezone.utc),
            cleaned_text="global warming leads to extreme climate events",
        ),
    ]

    for tweet in similar_tweets:
        cluster_id = engine.process_tweet(tweet)
        print(
            f"Processed duplicate-like tweet: {tweet.text[:50]}... → Cluster {cluster_id}"
        )

    # Check for new alerts (should be deduplicated)
    new_alert_candidates = engine.get_clusters_for_alerting()
    print(f"\nNew alert candidates after similar tweets: {len(new_alert_candidates)}")

    # Show deduplication statistics
    dedup_stats = engine.get_statistics()
    print(f"Alerts generated: {dedup_stats['alerts_generated']}")
    print(f"Alerts deduplicated: {dedup_stats['alerts_deduplicated']}")

    print("\n✅ Test completed successfully!")

    # Cleanup
    engine.cleanup()


def test_window_expiry():
    """Test natural window expiry behavior"""
    print("\n🕐 Testing Window Expiry Behavior")
    print("=" * 40)

    # Create engine with very short windows for testing
    engine = SlidingWindowClusteringEngine(
        window_size_hours=0.01, window_overlap_hours=0.005  # ~36 seconds  # ~18 seconds
    )

    # Create tweets with time gaps
    tweets = []
    base_time = datetime.now(timezone.utc)

    for i in range(10):
        tweet = Tweet(
            id=f"expiry_tweet_{i}",
            text=f"Test tweet number {i} for expiry testing",
            author=f"user_{i}",
            created_at=base_time + timedelta(seconds=i * 10),
            cleaned_text=f"test tweet number {i} for expiry testing",
        )
        tweets.append(tweet)

    print("Processing tweets with time gaps...")

    for i, tweet in enumerate(tweets):
        cluster_id = engine.process_tweet(tweet)
        stats = engine.get_statistics()

        print(
            f"Tweet {i+1}: {stats['active_windows']} windows, "
            f"{stats['active_clusters']} clusters, "
            f"{stats['windows_expired']} expired"
        )

        # Add delay to trigger window expiry
        if i % 3 == 2:  # Every 3rd tweet
            print("  ⏳ Waiting for window expiry...")
            time.sleep(2)  # Wait for windows to expire

    final_stats = engine.get_statistics()
    print(
        f"\nFinal: {final_stats['windows_created']} created, "
        f"{final_stats['windows_expired']} expired, "
        f"{final_stats['clusters_expired']} clusters expired"
    )

    engine.cleanup()


if __name__ == "__main__":
    # Run main clustering test
    test_sliding_window_clustering()

    # Run window expiry test
    test_window_expiry()

    print("\n🎉 All tests completed!")
