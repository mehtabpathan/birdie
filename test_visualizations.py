#!/usr/bin/env python3
"""
Test script to generate comprehensive visualizations of BIRCH clustering results
"""

import json
import sys
from pathlib import Path
from typing import Dict

from loguru import logger

from cluster_visualizer import ClusterVisualizer
from dummy_data_generator import DummyTweetGenerator
from models import Tweet
from tweet_pipeline import TweetProcessingPipeline


def load_dummy_tweets() -> Dict[str, Tweet]:
    """Load dummy tweets from JSON file"""
    tweets_file = Path("dummy_tweets.json")

    if not tweets_file.exists():
        logger.info("Dummy tweets file not found, generating new tweets...")
        generator = DummyTweetGenerator()
        tweets_list = generator.generate_dummy_tweets(
            1000
        )  # Generate 1000 tweets for better visualization
        generator.save_tweets_to_json(tweets_list)

    with open(tweets_file, "r", encoding="utf-8") as f:
        tweets_data = json.load(f)

    tweets = {}
    for tweet_data in tweets_data:
        tweet = Tweet(
            tweet_id=tweet_data["tweet_id"],
            text=tweet_data["text"],
            created_at=tweet_data["created_at"],
            author_id=tweet_data["author_id"],
            author_username=tweet_data["author_username"],
        )
        tweets[tweet.id] = tweet

    logger.info(f"Loaded {len(tweets)} dummy tweets")
    return tweets


def run_clustering_for_visualization():
    """Run clustering pipeline and generate visualizations"""
    logger.info("🚀 Starting clustering visualization test...")

    # Load tweets
    tweets = load_dummy_tweets()

    # Initialize pipeline
    pipeline = TweetProcessingPipeline()

    # Process tweets in batches for better clustering
    tweet_list = list(tweets.values())
    batch_size = 50

    logger.info(f"Processing {len(tweet_list)} tweets in batches of {batch_size}")

    for i in range(0, len(tweet_list), batch_size):
        batch = tweet_list[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(tweet_list) + batch_size - 1)//batch_size}"
        )

        result = pipeline.add_tweets_batch(batch)
        logger.info(f"Batch result: {result}")

    # Get final statistics
    stats = pipeline.clustering_engine.get_statistics()
    logger.info(f"Final clustering statistics: {stats}")

    # Create visualizer
    visualizer = ClusterVisualizer(pipeline.clustering_engine, tweets)

    # Print text summary first
    visualizer.print_cluster_summary()

    # Generate all visualizations
    logger.info("🎨 Generating visualizations...")

    try:
        # 1. Comprehensive Dashboard
        logger.info("Creating comprehensive dashboard...")
        dashboard_fig = visualizer.create_comprehensive_dashboard(
            "cluster_dashboard.html"
        )

        # 2. Quality Report
        logger.info("Creating quality report...")
        quality_fig = visualizer.create_cluster_quality_report(
            "cluster_quality_report.html"
        )

        # 3. Topic Analysis
        logger.info("Creating topic analysis...")
        topic_fig = visualizer.create_topic_analysis("topic_analysis.html")

        # 4. Static Summary Plots
        logger.info("Creating static summary plots...")
        visualizer.create_static_summary_plots()

        logger.success("✅ All visualizations created successfully!")

        print("\n" + "=" * 60)
        print("📊 VISUALIZATION FILES CREATED:")
        print("=" * 60)
        print("🌐 Interactive Dashboards:")
        print("   • cluster_dashboard.html - Comprehensive clustering overview")
        print("   • cluster_quality_report.html - Detailed quality analysis")
        print("   • topic_analysis.html - Topic-focused insights")
        print("\n📈 Static Plots:")
        print("   • clustering_summary.png - Summary charts")
        print(
            "\n💡 Open the HTML files in your browser to explore interactive visualizations!"
        )
        print("=" * 60)

    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main function"""
    try:
        run_clustering_for_visualization()
    except KeyboardInterrupt:
        logger.info("Visualization test interrupted by user")
    except Exception as e:
        logger.error(f"Visualization test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
