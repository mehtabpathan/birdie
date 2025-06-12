import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import config
from models import Cluster, ClusterStatus, TopicOfInterest, Tweet
from text_processor import TextProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IncrementalClusteringEngine:
    def __init__(self):
        self.text_processor = TextProcessor()

        # Initialize sentence transformer model
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.embedding_dimension = (
            self.embedding_model.get_sentence_embedding_dimension()
        )

        # Initialize FAISS index for efficient similarity search
        self.index = faiss.IndexFlatIP(
            self.embedding_dimension
        )  # Inner product for cosine similarity

        # Store cluster data
        self.clusters: Dict[str, Cluster] = {}
        self.cluster_embeddings: Dict[str, np.ndarray] = {}
        self.cluster_id_to_index: Dict[str, int] = {}
        self.index_to_cluster_id: Dict[int, str] = {}

        # Topics of interest
        self.topics_of_interest = self._initialize_topics()

        # Statistics
        self.stats = {
            "tweets_processed": 0,
            "clusters_created": 0,
            "clusters_merged": 0,
            "tweets_clustered": 0,
        }

    def _initialize_topics(self) -> List[TopicOfInterest]:
        """Initialize topics of interest with embeddings"""
        topics = []

        for topic_config in config.TOPICS_OF_INTEREST:
            # Create topic description for embedding
            topic_text = f"{topic_config['title']} {topic_config['description']}"

            # Generate embedding
            embedding = self.embedding_model.encode([topic_text])[0]

            topic = TopicOfInterest(
                title=topic_config["title"],
                description=topic_config["description"],
                embedding=embedding.tolist(),
            )
            topics.append(topic)

        logger.info(f"Initialized {len(topics)} topics of interest")
        return topics

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not text or not text.strip():
            return np.zeros(self.embedding_dimension)

        try:
            embedding = self.embedding_model.encode([text])[0]
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dimension)

    def _find_similar_clusters(
        self, tweet_embedding: np.ndarray, threshold: float = None
    ) -> List[Tuple[str, float]]:
        """Find similar clusters using FAISS"""
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD

        if self.index.ntotal == 0:
            return []

        # Search for similar clusters
        tweet_embedding = tweet_embedding.reshape(1, -1).astype("float32")
        similarities, indices = self.index.search(
            tweet_embedding, min(10, self.index.ntotal)
        )

        similar_clusters = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx != -1 and similarity >= threshold:
                cluster_id = self.index_to_cluster_id[idx]
                similar_clusters.append((cluster_id, float(similarity)))

        return similar_clusters

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

    def _create_new_cluster(self, tweet: Tweet, tweet_embedding: np.ndarray) -> Cluster:
        """Create a new cluster with the tweet"""
        cluster = Cluster()
        cluster.add_tweet(tweet.id)
        cluster.centroid = tweet_embedding.tolist()

        # Check topic similarity
        topic_match, topic_similarity = self._calculate_topic_similarity(
            tweet_embedding
        )
        cluster.topic_match = topic_match
        cluster.topic_similarity = topic_similarity

        # Add to FAISS index
        cluster_idx = self.index.ntotal
        self.index.add(tweet_embedding.reshape(1, -1).astype("float32"))

        # Update mappings
        self.clusters[cluster.id] = cluster
        self.cluster_embeddings[cluster.id] = tweet_embedding
        self.cluster_id_to_index[cluster.id] = cluster_idx
        self.index_to_cluster_id[cluster_idx] = cluster.id

        # Update tweet
        tweet.cluster_id = cluster.id

        self.stats["clusters_created"] += 1
        logger.info(
            f"Created new cluster {cluster.id} with topic match: {topic_match} (similarity: {topic_similarity:.3f})"
        )

        return cluster

    def _add_to_cluster(
        self, tweet: Tweet, cluster_id: str, tweet_embedding: np.ndarray
    ):
        """Add tweet to existing cluster and update centroid"""
        cluster = self.clusters[cluster_id]
        cluster.add_tweet(tweet.id)

        # Update centroid (running average)
        old_centroid = self.cluster_embeddings[cluster_id]
        new_centroid = (
            old_centroid * (cluster.size - 1) + tweet_embedding
        ) / cluster.size
        new_centroid = new_centroid / np.linalg.norm(new_centroid)  # Normalize

        # Update stored data
        cluster.centroid = new_centroid.tolist()
        self.cluster_embeddings[cluster_id] = new_centroid

        # Update FAISS index
        cluster_idx = self.cluster_id_to_index[cluster_id]
        self.index.remove_ids(np.array([cluster_idx]))
        self.index.add(new_centroid.reshape(1, -1).astype("float32"))

        # Update topic similarity
        topic_match, topic_similarity = self._calculate_topic_similarity(new_centroid)
        cluster.topic_match = topic_match
        cluster.topic_similarity = topic_similarity

        # Update tweet
        tweet.cluster_id = cluster_id

        self.stats["tweets_clustered"] += 1
        logger.debug(
            f"Added tweet {tweet.id} to cluster {cluster_id} (size: {cluster.size})"
        )

    def process_tweet(self, tweet: Tweet) -> Optional[str]:
        """Process a single tweet for clustering"""
        self.stats["tweets_processed"] += 1

        # Clean and process the tweet
        processed_tweet = self.text_processor.process_tweet(tweet)

        # Check if tweet has meaningful content
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

        # Find similar clusters
        similar_clusters = self._find_similar_clusters(tweet_embedding)

        if similar_clusters:
            # Add to most similar cluster
            best_cluster_id, similarity = similar_clusters[0]
            self._add_to_cluster(tweet, best_cluster_id, tweet_embedding)
            return best_cluster_id
        else:
            # Create new cluster
            cluster = self._create_new_cluster(tweet, tweet_embedding)
            return cluster.id

    def process_tweets_batch(self, tweets: List[Tweet]) -> Dict[str, List[str]]:
        """Process a batch of tweets"""
        results = {"clustered": [], "new_clusters": [], "skipped": []}

        logger.info(f"Processing batch of {len(tweets)} tweets")

        for tweet in tweets:
            cluster_id = self.process_tweet(tweet)

            if cluster_id:
                if cluster_id in [c.id for c in self.clusters.values() if c.size == 1]:
                    results["new_clusters"].append(cluster_id)
                else:
                    results["clustered"].append(cluster_id)
            else:
                results["skipped"].append(tweet.id)

        logger.info(
            f"Batch processing complete: {len(results['clustered'])} clustered, "
            f"{len(results['new_clusters'])} new clusters, {len(results['skipped'])} skipped"
        )

        return results

    def get_clusters_for_alerting(self) -> List[Cluster]:
        """Get clusters that meet alerting criteria"""
        alert_clusters = []

        for cluster in self.clusters.values():
            if (
                cluster.status == ClusterStatus.ACTIVE
                and not cluster.alert_sent
                and cluster.size >= config.MIN_CLUSTER_SIZE
                and cluster.topic_similarity >= 0.6
            ):  # Minimum topic similarity threshold

                alert_clusters.append(cluster)

        return alert_clusters

    def expire_old_clusters(self) -> List[str]:
        """Expire clusters that are too old"""
        expired_cluster_ids = []

        for cluster_id, cluster in list(self.clusters.items()):
            if cluster.is_expired(config.CLUSTER_EXPIRY_HOURS):
                cluster.status = ClusterStatus.EXPIRED
                expired_cluster_ids.append(cluster_id)

                # Remove from FAISS index
                if cluster_id in self.cluster_id_to_index:
                    cluster_idx = self.cluster_id_to_index[cluster_id]
                    # Note: FAISS doesn't support efficient removal, so we'll rebuild periodically

                logger.info(
                    f"Expired cluster {cluster_id} (last tweet: {cluster.last_tweet_added})"
                )

        return expired_cluster_ids

    def get_cluster_summary(self, cluster_id: str, tweets: Dict[str, Tweet]) -> str:
        """Generate a summary for a cluster"""
        if cluster_id not in self.clusters:
            return ""

        cluster = self.clusters[cluster_id]

        # Get tweets in cluster
        cluster_tweets = [
            tweets[tweet_id] for tweet_id in cluster.tweet_ids if tweet_id in tweets
        ]

        if not cluster_tweets:
            return ""

        # Simple summary: most common words
        all_text = " ".join([tweet.cleaned_text for tweet in cluster_tweets])
        words = all_text.split()

        # Count word frequency
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        summary = f"Cluster about: {', '.join([word for word, _ in top_words])}"

        if cluster.topic_match:
            summary += f" (Topic: {cluster.topic_match})"

        return summary

    def get_statistics(self) -> Dict:
        """Get clustering statistics"""
        active_clusters = len(
            [c for c in self.clusters.values() if c.status == ClusterStatus.ACTIVE]
        )
        expired_clusters = len(
            [c for c in self.clusters.values() if c.status == ClusterStatus.EXPIRED]
        )

        return {
            **self.stats,
            "active_clusters": active_clusters,
            "expired_clusters": expired_clusters,
            "total_clusters": len(self.clusters),
            "average_cluster_size": (
                np.mean([c.size for c in self.clusters.values()])
                if self.clusters
                else 0
            ),
        }

    def rebuild_index(self):
        """Rebuild FAISS index (useful after many deletions)"""
        logger.info("Rebuilding FAISS index...")

        # Create new index
        new_index = faiss.IndexFlatIP(self.embedding_dimension)
        new_cluster_id_to_index = {}
        new_index_to_cluster_id = {}

        # Add active clusters
        idx = 0
        for cluster_id, cluster in self.clusters.items():
            if cluster.status == ClusterStatus.ACTIVE:
                embedding = self.cluster_embeddings[cluster_id]
                new_index.add(embedding.reshape(1, -1).astype("float32"))
                new_cluster_id_to_index[cluster_id] = idx
                new_index_to_cluster_id[idx] = cluster_id
                idx += 1

        # Update references
        self.index = new_index
        self.cluster_id_to_index = new_cluster_id_to_index
        self.index_to_cluster_id = new_index_to_cluster_id

        logger.info(f"Index rebuilt with {self.index.ntotal} active clusters")
