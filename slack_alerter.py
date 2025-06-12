import json
import logging
from datetime import datetime
from typing import Dict, List

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from config import config
from models import Cluster, ClusterStatus, Tweet

logger = logging.getLogger(__name__)


class SlackAlerter:
    def __init__(self):
        if not config.SLACK_BOT_TOKEN:
            logger.warning(
                "Slack bot token not configured. Alerts will be logged only."
            )
            self.client = None
        else:
            self.client = WebClient(token=config.SLACK_BOT_TOKEN)
            self._test_connection()

    def _test_connection(self):
        """Test Slack connection"""
        try:
            response = self.client.auth_test()
            logger.info(f"Connected to Slack as {response['user']}")
        except SlackApiError as e:
            logger.error(f"Failed to connect to Slack: {e}")
            self.client = None

    def _format_cluster_alert(self, cluster: Cluster, tweets: Dict[str, Tweet]) -> Dict:
        """Format cluster information for Slack alert"""
        # Get tweets in cluster
        cluster_tweets = [
            tweets[tweet_id] for tweet_id in cluster.tweet_ids if tweet_id in tweets
        ]

        # Get representative tweets (most liked/retweeted)
        representative_tweets = sorted(
            cluster_tweets, key=lambda t: t.like_count + t.retweet_count, reverse=True
        )[:3]

        # Calculate total engagement
        total_engagement = sum(
            tweet.like_count + tweet.retweet_count + tweet.reply_count
            for tweet in cluster_tweets
        )

        # Create alert message
        alert_text = f"🚨 *New Tweet Cluster Alert* 🚨\n\n"
        alert_text += f"*Topic:* {cluster.topic_match or 'General Discussion'}\n"
        alert_text += f"*Cluster Size:* {cluster.size} tweets\n"
        alert_text += f"*Topic Similarity:* {cluster.topic_similarity:.2%}\n"
        alert_text += f"*Total Engagement:* {total_engagement:,} interactions\n"
        alert_text += f"*Time Period:* {cluster.created_at.strftime('%Y-%m-%d %H:%M')} - {cluster.last_tweet_added.strftime('%Y-%m-%d %H:%M')}\n\n"

        # Add representative tweets
        alert_text += "*Representative Tweets:*\n"
        for i, tweet in enumerate(representative_tweets, 1):
            alert_text += (
                f"{i}. _{tweet.text[:100]}{'...' if len(tweet.text) > 100 else ''}_\n"
            )
            alert_text += f"   👤 @{tweet.author_username} | ❤️ {tweet.like_count} | 🔄 {tweet.retweet_count}\n\n"

        # Create blocks for rich formatting
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🚨 Tweet Cluster Alert"},
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Topic:*\n{cluster.topic_match or 'General Discussion'}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Cluster Size:*\n{cluster.size} tweets",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Topic Similarity:*\n{cluster.topic_similarity:.2%}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Total Engagement:*\n{total_engagement:,} interactions",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Time Period:* {cluster.created_at.strftime('%Y-%m-%d %H:%M')} - {cluster.last_tweet_added.strftime('%Y-%m-%d %H:%M')}",
                },
            },
            {"type": "divider"},
        ]

        # Add representative tweets as blocks
        for i, tweet in enumerate(representative_tweets, 1):
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Tweet {i}:*\n_{tweet.text[:200]}{'...' if len(tweet.text) > 200 else ''}_\n👤 @{tweet.author_username} | ❤️ {tweet.like_count} | 🔄 {tweet.retweet_count} | 💬 {tweet.reply_count}",
                    },
                }
            )

        return {"text": alert_text, "blocks": blocks}

    def send_cluster_alert(self, cluster: Cluster, tweets: Dict[str, Tweet]) -> bool:
        """Send alert for a cluster"""
        try:
            # Format the alert
            alert_data = self._format_cluster_alert(cluster, tweets)

            if self.client:
                # Send to Slack
                response = self.client.chat_postMessage(
                    channel=config.SLACK_CHANNEL,
                    text=alert_data["text"],
                    blocks=alert_data["blocks"],
                )

                if response["ok"]:
                    logger.info(f"Sent Slack alert for cluster {cluster.id}")
                    return True
                else:
                    logger.error(f"Failed to send Slack alert: {response}")
                    return False
            else:
                # Log the alert if Slack is not configured
                logger.info(f"ALERT (Slack not configured): {alert_data['text']}")
                return True

        except SlackApiError as e:
            logger.error(f"Slack API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending cluster alert: {e}")
            return False

    def send_summary_alert(self, stats: Dict) -> bool:
        """Send daily/periodic summary alert"""
        try:
            summary_text = f"📊 *Tweet Clustering Summary* 📊\n\n"
            summary_text += (
                f"*Tweets Processed:* {stats.get('tweets_processed', 0):,}\n"
            )
            summary_text += f"*Active Clusters:* {stats.get('active_clusters', 0)}\n"
            summary_text += f"*Clusters Created:* {stats.get('clusters_created', 0)}\n"
            summary_text += (
                f"*Tweets Clustered:* {stats.get('tweets_clustered', 0):,}\n"
            )
            summary_text += (
                f"*Average Cluster Size:* {stats.get('average_cluster_size', 0):.1f}\n"
            )
            summary_text += f"*Alerts Sent:* {stats.get('alerts_sent', 0)}\n"

            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "📊 Tweet Clustering Summary",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Tweets Processed:*\n{stats.get('tweets_processed', 0):,}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Active Clusters:*\n{stats.get('active_clusters', 0)}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Clusters Created:*\n{stats.get('clusters_created', 0)}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Tweets Clustered:*\n{stats.get('tweets_clustered', 0):,}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Average Cluster Size:*\n{stats.get('average_cluster_size', 0):.1f}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Alerts Sent:*\n{stats.get('alerts_sent', 0)}",
                        },
                    ],
                },
            ]

            if self.client:
                response = self.client.chat_postMessage(
                    channel=config.SLACK_CHANNEL, text=summary_text, blocks=blocks
                )

                if response["ok"]:
                    logger.info("Sent summary alert to Slack")
                    return True
                else:
                    logger.error(f"Failed to send summary alert: {response}")
                    return False
            else:
                logger.info(f"SUMMARY ALERT (Slack not configured): {summary_text}")
                return True

        except Exception as e:
            logger.error(f"Error sending summary alert: {e}")
            return False

    def send_error_alert(self, error_message: str, context: str = "") -> bool:
        """Send error alert"""
        try:
            alert_text = f"🚨 *Tweet Pipeline Error* 🚨\n\n"
            alert_text += f"*Error:* {error_message}\n"
            if context:
                alert_text += f"*Context:* {context}\n"
            alert_text += f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

            if self.client:
                response = self.client.chat_postMessage(
                    channel=config.SLACK_CHANNEL, text=alert_text
                )

                if response["ok"]:
                    logger.info("Sent error alert to Slack")
                    return True
                else:
                    logger.error(f"Failed to send error alert: {response}")
                    return False
            else:
                logger.error(f"ERROR ALERT (Slack not configured): {alert_text}")
                return True

        except Exception as e:
            logger.error(f"Error sending error alert: {e}")
            return False

    def test_alert(self) -> bool:
        """Send a test alert to verify configuration"""
        try:
            test_text = (
                "🧪 *Test Alert* 🧪\n\nTweet clustering pipeline is working correctly!"
            )

            if self.client:
                response = self.client.chat_postMessage(
                    channel=config.SLACK_CHANNEL, text=test_text
                )

                if response["ok"]:
                    logger.info("Test alert sent successfully")
                    return True
                else:
                    logger.error(f"Test alert failed: {response}")
                    return False
            else:
                logger.info(f"TEST ALERT (Slack not configured): {test_text}")
                return True

        except Exception as e:
            logger.error(f"Error sending test alert: {e}")
            return False
