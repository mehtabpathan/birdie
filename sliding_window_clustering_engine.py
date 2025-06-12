import hashlib
import json
import logging
import pickle
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import config
from models import Cluster, ClusterStatus, TopicOfInterest, Tweet
from text_processor import TextProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AlertRecord:
    """Record of a generated alert"""

    alert_id: str
    cluster_id: str
    topic: str
    keywords: List[str]
    embedding: List[float]
    timestamp: datetime
    tweet_count: int
    similarity_threshold_used: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "AlertRecord":
        """Create from dictionary"""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class WindowedTweet:
    """Tweet with window metadata"""

    tweet: Tweet
    embedding: np.ndarray
    window_id: int
    added_at: datetime


class SemanticAlertDeduplicator:
    """Handles semantic deduplication of alerts with persistence"""

    def __init__(self, similarity_threshold: float = 0.85, cooldown_hours: int = 24):
        self.similarity_threshold = similarity_threshold
        self.cooldown_hours = cooldown_hours
        self.alert_history: Dict[str, AlertRecord] = {}
        self.alert_embeddings: Dict[str, np.ndarray] = {}
        self.persistence_file = "alert_history.pkl"

        # Load previous alert history
        self._load_alert_history()

    def _load_alert_history(self):
        """Load alert history from disk"""
        try:
            with open(self.persistence_file, "rb") as f:
                data = pickle.load(f)
                self.alert_history = data.get("alert_history", {})
                self.alert_embeddings = data.get("alert_embeddings", {})

            # Clean up old alerts beyond cooldown period
            self._cleanup_old_alerts()

            logger.info(f"Loaded {len(self.alert_history)} alert records from disk")
        except FileNotFoundError:
            logger.info("No previous alert history found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading alert history: {e}")

    def _save_alert_history(self):
        """Save alert history to disk"""
        try:
            data = {
                "alert_history": self.alert_history,
                "alert_embeddings": self.alert_embeddings,
            }
            with open(self.persistence_file, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Saved {len(self.alert_history)} alert records to disk")
        except Exception as e:
            logger.error(f"Error saving alert history: {e}")

    def _cleanup_old_alerts(self):
        """Remove alerts older than cooldown period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.cooldown_hours)

        old_alert_ids = [
            alert_id
            for alert_id, alert in self.alert_history.items()
            if alert.timestamp < cutoff_time
        ]

        for alert_id in old_alert_ids:
            del self.alert_history[alert_id]
            if alert_id in self.alert_embeddings:
                del self.alert_embeddings[alert_id]

        if old_alert_ids:
            logger.info(f"Cleaned up {len(old_alert_ids)} old alert records")

    def _generate_alert_fingerprint(self, topic: str, keywords: List[str]) -> str:
        """Generate fingerprint for alert based on topic and keywords"""
        content = f"{topic}:{':'.join(sorted(keywords))}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _extract_keywords(
        self, cluster_tweets: List[Tweet], top_k: int = 5
    ) -> List[str]:
        """Extract top keywords from cluster tweets"""
        word_freq = defaultdict(int)

        for tweet in cluster_tweets:
            words = tweet.cleaned_text.lower().split()
            for word in words:
                if len(word) > 3 and word.isalpha():  # Filter meaningful words
                    word_freq[word] += 1

        # Get top keywords
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [word for word, _ in top_words]

    def should_generate_alert(
        self,
        cluster: Cluster,
        cluster_tweets: List[Tweet],
        cluster_embedding: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Determine if an alert should be generated based on semantic deduplication
        Returns (should_alert, reason)
        """
        # Extract cluster characteristics
        topic = cluster.topic_match or "general"
        keywords = self._extract_keywords(cluster_tweets)

        # Generate fingerprint
        fingerprint = self._generate_alert_fingerprint(topic, keywords)

        # Check for exact fingerprint match within cooldown
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.cooldown_hours)

        for alert_id, alert in self.alert_history.items():
            if alert.timestamp < cutoff_time:
                continue

            # Check fingerprint match
            alert_fingerprint = self._generate_alert_fingerprint(
                alert.topic, alert.keywords
            )
            if fingerprint == alert_fingerprint:
                return (
                    False,
                    f"Exact topic/keyword match with alert {alert_id} from {alert.timestamp}",
                )

        # Check semantic similarity with recent alerts
        if len(self.alert_embeddings) > 0:
            similarities = []
            recent_alert_ids = []

            for alert_id, alert in self.alert_history.items():
                if alert.timestamp < cutoff_time:
                    continue

                if alert_id in self.alert_embeddings:
                    alert_embedding = self.alert_embeddings[alert_id]
                    similarity = cosine_similarity(
                        [cluster_embedding], [alert_embedding]
                    )[0][0]
                    similarities.append(similarity)
                    recent_alert_ids.append(alert_id)

            if similarities:
                max_similarity = max(similarities)
                if max_similarity > self.similarity_threshold:
                    similar_alert_id = recent_alert_ids[
                        similarities.index(max_similarity)
                    ]
                    similar_alert = self.alert_history[similar_alert_id]
                    return (
                        False,
                        f"Semantic similarity {max_similarity:.3f} with alert {similar_alert_id} from {similar_alert.timestamp}",
                    )

        return True, "No similar recent alerts found"

    def record_alert(
        self,
        cluster: Cluster,
        cluster_tweets: List[Tweet],
        cluster_embedding: np.ndarray,
    ) -> str:
        """Record a generated alert"""
        topic = cluster.topic_match or "general"
        keywords = self._extract_keywords(cluster_tweets)

        alert_record = AlertRecord(
            alert_id=f"alert_{cluster.id}_{int(datetime.now().timestamp())}",
            cluster_id=cluster.id,
            topic=topic,
            keywords=keywords,
            embedding=cluster_embedding.tolist(),
            timestamp=datetime.now(timezone.utc),
            tweet_count=cluster.size,
            similarity_threshold_used=self.similarity_threshold,
        )

        # Store alert
        self.alert_history[alert_record.alert_id] = alert_record
        self.alert_embeddings[alert_record.alert_id] = cluster_embedding

        # Persist to disk
        self._save_alert_history()

        logger.info(f"Recorded alert {alert_record.alert_id} for cluster {cluster.id}")
        return alert_record.alert_id

    def get_alert_statistics(self) -> Dict:
        """Get statistics about alert history"""
        if not self.alert_history:
            return {"total_alerts": 0}

        # Group by topic
        topic_counts = defaultdict(int)
        for alert in self.alert_history.values():
            topic_counts[alert.topic] += 1

        # Recent alerts (last 24 hours)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_alerts = [
            alert
            for alert in self.alert_history.values()
            if alert.timestamp > recent_cutoff
        ]

        return {
            "total_alerts": len(self.alert_history),
            "recent_alerts_24h": len(recent_alerts),
            "topics_covered": len(topic_counts),
            "top_topics": dict(
                sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "oldest_alert": min(
                alert.timestamp for alert in self.alert_history.values()
            ).isoformat(),
            "newest_alert": max(
                alert.timestamp for alert in self.alert_history.values()
            ).isoformat(),
        }


class SlidingWindowClusteringEngine:
    """
    Sliding window clustering engine with natural expiry and semantic alert deduplication
    """

    def __init__(self, window_size_hours: int = 24, window_overlap_hours: int = 6):
        self.window_size_hours = window_size_hours
        self.window_overlap_hours = window_overlap_hours
        self.window_slide_hours = window_size_hours - window_overlap_hours

        # Initialize components
        self.text_processor = TextProcessor()

        # Initialize embedding model
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.embedding_dimension = (
            self.embedding_model.get_sentence_embedding_dimension()
        )

        # Sliding window storage
        self.windows: Dict[int, List[WindowedTweet]] = {}
        self.current_window_id = 0
        self.window_start_times: Dict[int, datetime] = {}

        # Active clusters (across all active windows)
        self.active_clusters: Dict[str, Cluster] = {}
        self.cluster_embeddings: Dict[str, np.ndarray] = {}
        self.cluster_tweets: Dict[str, List[Tweet]] = {}

        # FAISS index for efficient similarity search
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.index_to_cluster_id: List[str] = []

        # Topics of interest
        self.topics_of_interest = self._initialize_topics()

        # Alert deduplication
        self.alert_deduplicator = SemanticAlertDeduplicator(
            similarity_threshold=0.85, cooldown_hours=24
        )

        # Clustering parameters
        self.similarity_threshold = 0.7
        self.min_cluster_size = config.MIN_CLUSTER_SIZE

        # Statistics
        self.stats = {
            "tweets_processed": 0,
            "clusters_created": 0,
            "clusters_expired": 0,
            "windows_created": 0,
            "windows_expired": 0,
            "alerts_generated": 0,
            "alerts_deduplicated": 0,
        }

        # Initialize first window
        self._create_new_window()

    def _initialize_topics(self) -> List[TopicOfInterest]:
        """Initialize topics of interest with embeddings"""
        topics = []

        for topic_config in config.TOPICS_OF_INTEREST:
            topic_text = f"{topic_config['title']} {topic_config['description']}"
            embedding = self.embedding_model.encode([topic_text])[0]

            topic = TopicOfInterest(
                title=topic_config["title"],
                description=topic_config["description"],
                embedding=embedding.tolist(),
            )
            topics.append(topic)

        logger.info(f"Initialized {len(topics)} topics of interest")
        return topics

    def _create_new_window(self) -> int:
        """Create a new sliding window"""
        window_id = self.current_window_id
        self.windows[window_id] = []
        self.window_start_times[window_id] = datetime.now(timezone.utc)
        self.current_window_id += 1
        self.stats["windows_created"] += 1

        logger.info(f"Created new window {window_id}")
        return window_id

    def _expire_old_windows(self):
        """Remove windows that are beyond the sliding window"""
        current_time = datetime.now(timezone.utc)
        expired_windows = []

        for window_id, start_time in list(self.window_start_times.items()):
            window_age_hours = (current_time - start_time).total_seconds() / 3600

            if window_age_hours > self.window_size_hours:
                expired_windows.append(window_id)

        for window_id in expired_windows:
            self._expire_window(window_id)

    def _expire_window(self, window_id: int):
        """Expire a specific window and clean up associated clusters"""
        if window_id not in self.windows:
            return

        logger.info(f"Expiring window {window_id}")

        # Get tweets in this window
        window_tweets = self.windows[window_id]
        window_tweet_ids = {wt.tweet.id for wt in window_tweets}

        # Find clusters that will be affected
        affected_clusters = []
        for cluster_id, cluster in list(self.active_clusters.items()):
            cluster_tweet_ids = set(cluster.tweet_ids)
            overlap = cluster_tweet_ids.intersection(window_tweet_ids)

            if overlap:
                # Remove tweets from this window from the cluster
                cluster.tweet_ids = [
                    tid for tid in cluster.tweet_ids if tid not in window_tweet_ids
                ]

                # Update cluster tweets
                if cluster_id in self.cluster_tweets:
                    self.cluster_tweets[cluster_id] = [
                        tweet
                        for tweet in self.cluster_tweets[cluster_id]
                        if tweet.id not in window_tweet_ids
                    ]

                # If cluster becomes too small, expire it
                if len(cluster.tweet_ids) < self.min_cluster_size:
                    affected_clusters.append(cluster_id)
                else:
                    # Update cluster centroid
                    self._update_cluster_centroid(cluster_id)

        # Expire clusters that became too small
        for cluster_id in affected_clusters:
            self._expire_cluster(cluster_id)

        # Clean up window
        del self.windows[window_id]
        del self.window_start_times[window_id]
        self.stats["windows_expired"] += 1

    def _expire_cluster(self, cluster_id: str):
        """Expire a cluster and clean up associated data"""
        if cluster_id not in self.active_clusters:
            return

        cluster = self.active_clusters[cluster_id]
        cluster.status = ClusterStatus.EXPIRED

        # Remove from active clusters
        del self.active_clusters[cluster_id]

        # Clean up embeddings and tweets
        if cluster_id in self.cluster_embeddings:
            del self.cluster_embeddings[cluster_id]
        if cluster_id in self.cluster_tweets:
            del self.cluster_tweets[cluster_id]

        # Remove from FAISS index
        self._rebuild_faiss_index()

        self.stats["clusters_expired"] += 1
        logger.info(f"Expired cluster {cluster_id}")

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not text or not text.strip():
            return np.zeros(self.embedding_dimension)

        try:
            embedding = self.embedding_model.encode([text])[0]
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dimension)

    def _calculate_topic_similarity(
        self, tweet_embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """Calculate similarity to topics of interest"""
        max_similarity = 0.0
        best_topic = None

        for topic in self.topics_of_interest:
            if topic.embedding:
                topic_embedding = np.array(topic.embedding)
                topic_embedding = topic_embedding / np.linalg.norm(topic_embedding)

                similarity = np.dot(tweet_embedding, topic_embedding)

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_topic = topic.title

        return best_topic, max_similarity

    def _find_similar_cluster(self, tweet_embedding: np.ndarray) -> Optional[str]:
        """Find the most similar cluster using FAISS"""
        if self.index.ntotal == 0:
            return None

        # Search for similar clusters
        similarities, indices = self.index.search(
            tweet_embedding.reshape(1, -1).astype(np.float32),
            k=min(5, self.index.ntotal),
        )

        # Check if any cluster is similar enough
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity > self.similarity_threshold:
                return self.index_to_cluster_id[idx]

        return None

    def _update_cluster_centroid(self, cluster_id: str):
        """Update cluster centroid based on current tweets"""
        if cluster_id not in self.cluster_tweets or not self.cluster_tweets[cluster_id]:
            return

        cluster_tweet_texts = [
            tweet.cleaned_text for tweet in self.cluster_tweets[cluster_id]
        ]
        if not cluster_tweet_texts:
            return

        # Calculate centroid as mean of tweet embeddings
        embeddings = []
        for text in cluster_tweet_texts:
            embedding = self._generate_embedding(text)
            if not np.allclose(embedding, 0):
                embeddings.append(embedding)

        if embeddings:
            centroid = np.mean(embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            self.cluster_embeddings[cluster_id] = centroid

            # Update cluster
            cluster = self.active_clusters[cluster_id]
            cluster.centroid = centroid.tolist()

            # Update topic similarity
            topic_match, topic_similarity = self._calculate_topic_similarity(centroid)
            cluster.topic_match = topic_match
            cluster.topic_similarity = topic_similarity

            # Rebuild FAISS index
            self._rebuild_faiss_index()

    def _rebuild_faiss_index(self):
        """Rebuild FAISS index with current cluster centroids"""
        if not self.cluster_embeddings:
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.index_to_cluster_id = []
            return

        # Create new index
        embeddings = []
        cluster_ids = []

        for cluster_id, embedding in self.cluster_embeddings.items():
            embeddings.append(embedding)
            cluster_ids.append(cluster_id)

        if embeddings:
            embeddings_matrix = np.array(embeddings).astype(np.float32)

            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.index.add(embeddings_matrix)
            self.index_to_cluster_id = cluster_ids

    def _should_slide_window(self) -> bool:
        """Check if it's time to create a new window"""
        if not self.window_start_times:
            return True

        current_time = datetime.now(timezone.utc)
        latest_window_start = max(self.window_start_times.values())
        hours_since_latest = (current_time - latest_window_start).total_seconds() / 3600

        return hours_since_latest >= self.window_slide_hours

    def process_tweet(self, tweet: Tweet) -> Optional[str]:
        """Process a single tweet with sliding window clustering"""
        self.stats["tweets_processed"] += 1

        # Clean and process tweet
        processed_tweet = self.text_processor.process_tweet(tweet)

        if not self.text_processor.is_meaningful_text(processed_tweet.cleaned_text):
            logger.debug(f"Tweet {tweet.id} has no meaningful content, skipping")
            return None

        # Generate embedding
        tweet_embedding = self._generate_embedding(processed_tweet.cleaned_text)

        if np.allclose(tweet_embedding, 0):
            logger.debug(f"Tweet {tweet.id} has zero embedding, skipping")
            return None

        # Store embedding in tweet
        tweet.embedding = tweet_embedding.tolist()

        # Check if we need to slide the window
        if self._should_slide_window():
            self._create_new_window()

        # Expire old windows
        self._expire_old_windows()

        # Add tweet to current window
        current_window = max(self.windows.keys())
        windowed_tweet = WindowedTweet(
            tweet=tweet,
            embedding=tweet_embedding,
            window_id=current_window,
            added_at=datetime.now(timezone.utc),
        )
        self.windows[current_window].append(windowed_tweet)

        # Find similar cluster
        similar_cluster_id = self._find_similar_cluster(tweet_embedding)

        if similar_cluster_id:
            # Add to existing cluster
            cluster = self.active_clusters[similar_cluster_id]
            cluster.add_tweet(tweet.id)

            if similar_cluster_id not in self.cluster_tweets:
                self.cluster_tweets[similar_cluster_id] = []
            self.cluster_tweets[similar_cluster_id].append(tweet)

            # Update cluster centroid
            self._update_cluster_centroid(similar_cluster_id)

            tweet.cluster_id = similar_cluster_id

            logger.debug(
                f"Added tweet {tweet.id} to existing cluster {similar_cluster_id}"
            )
            return similar_cluster_id

        else:
            # Create new cluster
            cluster = Cluster()
            cluster.add_tweet(tweet.id)
            cluster.centroid = tweet_embedding.tolist()

            # Calculate topic similarity
            topic_match, topic_similarity = self._calculate_topic_similarity(
                tweet_embedding
            )
            cluster.topic_match = topic_match
            cluster.topic_similarity = topic_similarity

            # Store cluster data
            self.active_clusters[cluster.id] = cluster
            self.cluster_embeddings[cluster.id] = tweet_embedding
            self.cluster_tweets[cluster.id] = [tweet]

            # Update FAISS index
            self._rebuild_faiss_index()

            tweet.cluster_id = cluster.id

            self.stats["clusters_created"] += 1
            logger.info(f"Created new cluster {cluster.id} with topic: {topic_match}")

            return cluster.id

    def get_clusters_for_alerting(self) -> List[Tuple[Cluster, List[Tweet], str]]:
        """Get clusters that meet alerting criteria with deduplication"""
        alert_candidates = []

        for cluster_id, cluster in self.active_clusters.items():
            if (
                cluster.status == ClusterStatus.ACTIVE
                and not cluster.alert_sent
                and cluster.size >= self.min_cluster_size
                and cluster.topic_similarity >= 0.6
            ):
                cluster_tweets = self.cluster_tweets.get(cluster_id, [])
                cluster_embedding = self.cluster_embeddings.get(cluster_id)

                if cluster_embedding is not None and cluster_tweets:
                    # Check if alert should be generated (semantic deduplication)
                    should_alert, reason = (
                        self.alert_deduplicator.should_generate_alert(
                            cluster, cluster_tweets, cluster_embedding
                        )
                    )

                    if should_alert:
                        alert_candidates.append((cluster, cluster_tweets, "New alert"))

                        # Record the alert
                        alert_id = self.alert_deduplicator.record_alert(
                            cluster, cluster_tweets, cluster_embedding
                        )
                        cluster.alert_sent = True
                        self.stats["alerts_generated"] += 1

                        logger.info(
                            f"Generated alert {alert_id} for cluster {cluster_id}"
                        )
                    else:
                        self.stats["alerts_deduplicated"] += 1
                        logger.info(
                            f"Deduplicated alert for cluster {cluster_id}: {reason}"
                        )

        return alert_candidates

    def get_statistics(self) -> Dict:
        """Get comprehensive clustering statistics"""
        # Basic cluster stats
        active_clusters = len(self.active_clusters)
        cluster_sizes = [c.size for c in self.active_clusters.values()]

        # Window stats
        active_windows = len(self.windows)
        total_tweets_in_windows = sum(len(tweets) for tweets in self.windows.values())

        # Alert stats
        alert_stats = self.alert_deduplicator.get_alert_statistics()

        return {
            **self.stats,
            "active_clusters": active_clusters,
            "active_windows": active_windows,
            "total_tweets_in_windows": total_tweets_in_windows,
            "average_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "faiss_index_size": self.index.ntotal,
            "window_size_hours": self.window_size_hours,
            "window_overlap_hours": self.window_overlap_hours,
            **alert_stats,
        }

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up sliding window clustering engine")
        # Alert deduplicator automatically saves state
        pass
