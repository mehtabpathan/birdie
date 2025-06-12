import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import Birch
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

        # Initialize BIRCH clustering for incremental clustering
        self.birch_clusterer = Birch(
            n_clusters=None,  # Let BIRCH determine the number of clusters
            threshold=0.3,  # Distance threshold for creating new clusters
            branching_factor=50,  # Maximum number of CF subclusters in each node
            compute_labels=True,
        )

        # Track if BIRCH has been fitted
        self.birch_fitted = False

        # Initialize FAISS index for efficient similarity search (for topic matching)
        self.index = faiss.IndexFlatIP(
            self.embedding_dimension
        )  # Inner product for cosine similarity

        # Store cluster data
        self.clusters: Dict[str, Cluster] = {}
        self.cluster_embeddings: Dict[str, np.ndarray] = {}
        self.cluster_id_to_birch_label: Dict[str, int] = {}
        self.birch_label_to_cluster_id: Dict[int, str] = {}

        # Store tweet embeddings and their BIRCH labels
        self.tweet_embeddings: Dict[str, np.ndarray] = {}
        self.tweet_birch_labels: Dict[str, int] = {}

        # Topics of interest
        self.topics_of_interest = self._initialize_topics()

        # Clustering parameters
        self.min_cluster_coherence = 0.6  # Minimum intra-cluster similarity
        self.quality_check_interval = 100  # Check cluster quality every N tweets
        self.birch_refit_interval = 1000  # Refit BIRCH every N tweets for optimization

        # Statistics
        self.stats = {
            "tweets_processed": 0,
            "clusters_created": 0,
            "clusters_merged": 0,
            "tweets_clustered": 0,
            "quality_checks_performed": 0,
            "birch_refits": 0,
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

    def _calculate_cluster_coherence(self, cluster_id: str) -> float:
        """Calculate intra-cluster coherence (average pairwise similarity)"""
        cluster = self.clusters[cluster_id]

        if cluster.size < 2:
            return 1.0

        # Get embeddings for tweets in cluster
        tweet_embeddings = []
        for tweet_id in cluster.tweet_ids:
            if tweet_id in self.tweet_embeddings:
                tweet_embeddings.append(self.tweet_embeddings[tweet_id])

        if len(tweet_embeddings) < 2:
            return 1.0

        # Calculate pairwise similarities
        embeddings_matrix = np.array(tweet_embeddings)
        similarities = cosine_similarity(embeddings_matrix)

        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu(similarities, k=1)
        non_zero_similarities = upper_triangle[upper_triangle > 0]

        return np.mean(non_zero_similarities) if len(non_zero_similarities) > 0 else 0.0

    def _update_cluster_centroid(self, cluster_id: str):
        """Update cluster centroid based on all tweets in the cluster"""
        cluster = self.clusters[cluster_id]

        # Get embeddings for all tweets in cluster
        tweet_embeddings = []
        for tweet_id in cluster.tweet_ids:
            if tweet_id in self.tweet_embeddings:
                tweet_embeddings.append(self.tweet_embeddings[tweet_id])

        if tweet_embeddings:
            # Calculate centroid as mean of all tweet embeddings
            centroid = np.mean(tweet_embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # Normalize

            cluster.centroid = centroid.tolist()
            self.cluster_embeddings[cluster_id] = centroid

            # Update topic similarity
            topic_match, topic_similarity = self._calculate_topic_similarity(centroid)
            cluster.topic_match = topic_match
            cluster.topic_similarity = topic_similarity

    def _create_or_update_cluster_from_birch_label(
        self, birch_label: int, tweet: Tweet, tweet_embedding: np.ndarray
    ):
        """Create a new cluster or update existing cluster based on BIRCH label"""

        if birch_label in self.birch_label_to_cluster_id:
            # Add to existing cluster
            cluster_id = self.birch_label_to_cluster_id[birch_label]
            cluster = self.clusters[cluster_id]
            cluster.add_tweet(tweet.id)

            # Store tweet embedding and label
            self.tweet_embeddings[tweet.id] = tweet_embedding
            self.tweet_birch_labels[tweet.id] = birch_label

            # Update cluster centroid
            self._update_cluster_centroid(cluster_id)

            # Update tweet
            tweet.cluster_id = cluster_id

            self.stats["tweets_clustered"] += 1
            logger.debug(
                f"Added tweet {tweet.id} to existing cluster {cluster_id} (size: {cluster.size})"
            )

            return cluster_id

        else:
            # Create new cluster
            cluster = Cluster()
            cluster.add_tweet(tweet.id)

            # Store tweet embedding and label
            self.tweet_embeddings[tweet.id] = tweet_embedding
            self.tweet_birch_labels[tweet.id] = birch_label

            # Set initial centroid
            cluster.centroid = tweet_embedding.tolist()
            self.cluster_embeddings[cluster.id] = tweet_embedding

            # Calculate topic similarity
            topic_match, topic_similarity = self._calculate_topic_similarity(
                tweet_embedding
            )
            cluster.topic_match = topic_match
            cluster.topic_similarity = topic_similarity

            # Update mappings
            self.clusters[cluster.id] = cluster
            self.cluster_id_to_birch_label[cluster.id] = birch_label
            self.birch_label_to_cluster_id[birch_label] = cluster.id

            # Update tweet
            tweet.cluster_id = cluster.id

            self.stats["clusters_created"] += 1
            logger.info(
                f"Created new cluster {cluster.id} from BIRCH label {birch_label} "
                f"with topic match: {topic_match} (similarity: {topic_similarity:.3f})"
            )

            return cluster.id

    def _perform_quality_maintenance(self):
        """Perform cluster quality maintenance"""
        self.stats["quality_checks_performed"] += 1

        # Check cluster coherence and log warnings for low-coherence clusters
        low_coherence_clusters = []
        for cluster_id in self.clusters.keys():
            cluster = self.clusters[cluster_id]
            if cluster.status == ClusterStatus.ACTIVE and cluster.size > 5:
                coherence = self._calculate_cluster_coherence(cluster_id)
                if coherence < self.min_cluster_coherence:
                    low_coherence_clusters.append((cluster_id, coherence))

        if low_coherence_clusters:
            logger.warning(
                f"Found {len(low_coherence_clusters)} clusters with low coherence"
            )
            for cluster_id, coherence in low_coherence_clusters[:3]:  # Log first 3
                logger.warning(f"Cluster {cluster_id} has coherence: {coherence:.3f}")

        logger.info(
            f"Quality maintenance completed. Checked {len(self.clusters)} clusters"
        )

    def _refit_birch_if_needed(self):
        """Refit BIRCH clustering periodically for optimization"""
        if (
            self.stats["tweets_processed"] % self.birch_refit_interval == 0
            and len(self.tweet_embeddings) > 100
        ):
            logger.info("Refitting BIRCH clustering for optimization...")

            # Get all tweet embeddings
            all_embeddings = list(self.tweet_embeddings.values())

            if len(all_embeddings) > 0:
                # Create new BIRCH clusterer with updated parameters
                new_birch = Birch(
                    n_clusters=None,
                    threshold=max(
                        0.2, 0.5 - len(all_embeddings) / 10000
                    ),  # Adaptive threshold
                    branching_factor=50,
                    compute_labels=True,
                )

                try:
                    # Fit on all embeddings
                    new_birch.fit(all_embeddings)

                    # Update the clusterer
                    self.birch_clusterer = new_birch
                    self.birch_fitted = True
                    self.stats["birch_refits"] += 1

                    logger.info(f"BIRCH refitted with {len(all_embeddings)} embeddings")

                except Exception as e:
                    logger.error(f"Error refitting BIRCH: {e}")

    def process_tweet(self, tweet: Tweet) -> Optional[str]:
        """Process a single tweet for clustering using BIRCH"""
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

        try:
            if not self.birch_fitted:
                # First tweet - initialize BIRCH
                self.birch_clusterer.fit([tweet_embedding])
                self.birch_fitted = True
                birch_label = 0
            else:
                # Use partial_fit for incremental learning
                self.birch_clusterer.partial_fit([tweet_embedding])
                # Predict cluster label for this tweet
                birch_label = self.birch_clusterer.predict([tweet_embedding])[0]

            # Create or update cluster based on BIRCH label
            cluster_id = self._create_or_update_cluster_from_birch_label(
                birch_label, tweet, tweet_embedding
            )

            # Perform periodic maintenance
            if self.stats["tweets_processed"] % self.quality_check_interval == 0:
                self._perform_quality_maintenance()

            # Refit BIRCH periodically for optimization
            self._refit_birch_if_needed()

            return cluster_id

        except Exception as e:
            logger.error(f"Error processing tweet {tweet.id} with BIRCH: {e}")
            return None

    def process_tweets_batch(self, tweets: List[Tweet]) -> Dict[str, List[str]]:
        """Process a batch of tweets with BIRCH clustering"""
        results = {"clustered": [], "new_clusters": [], "skipped": []}

        logger.info(f"Processing batch of {len(tweets)} tweets with BIRCH")

        for tweet in tweets:
            cluster_id = self.process_tweet(tweet)

            if cluster_id:
                cluster = self.clusters[cluster_id]
                if cluster.size == 1:  # New cluster
                    results["new_clusters"].append(cluster_id)
                else:  # Added to existing cluster
                    results["clustered"].append(cluster_id)
            else:
                results["skipped"].append(tweet.id)

        logger.info(
            f"BIRCH batch processing complete: {len(results['clustered'])} clustered, "
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

                # Clean up tweet embeddings and labels
                for tweet_id in cluster.tweet_ids:
                    if tweet_id in self.tweet_embeddings:
                        del self.tweet_embeddings[tweet_id]
                    if tweet_id in self.tweet_birch_labels:
                        del self.tweet_birch_labels[tweet_id]

                # Clean up cluster mappings
                if cluster_id in self.cluster_id_to_birch_label:
                    birch_label = self.cluster_id_to_birch_label[cluster_id]
                    del self.cluster_id_to_birch_label[cluster_id]
                    if birch_label in self.birch_label_to_cluster_id:
                        del self.birch_label_to_cluster_id[birch_label]

                if cluster_id in self.cluster_embeddings:
                    del self.cluster_embeddings[cluster_id]

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

        # Add coherence information
        coherence = self._calculate_cluster_coherence(cluster_id)
        summary += f" [Coherence: {coherence:.2f}]"

        # Add BIRCH label information
        if cluster_id in self.cluster_id_to_birch_label:
            birch_label = self.cluster_id_to_birch_label[cluster_id]
            summary += f" [BIRCH: {birch_label}]"

        return summary

    def get_statistics(self) -> Dict:
        """Get clustering statistics"""
        active_clusters = len(
            [c for c in self.clusters.values() if c.status == ClusterStatus.ACTIVE]
        )
        expired_clusters = len(
            [c for c in self.clusters.values() if c.status == ClusterStatus.EXPIRED]
        )

        # Calculate cluster size distribution
        cluster_sizes = [
            c.size for c in self.clusters.values() if c.status == ClusterStatus.ACTIVE
        ]
        large_clusters = len([s for s in cluster_sizes if s > 50])

        # Calculate average coherence
        coherences = []
        for cluster_id in list(self.clusters.keys())[
            :10
        ]:  # Sample first 10 for performance
            if self.clusters[cluster_id].status == ClusterStatus.ACTIVE:
                coherences.append(self._calculate_cluster_coherence(cluster_id))

        avg_coherence = np.mean(coherences) if coherences else 0.0

        return {
            **self.stats,
            "active_clusters": active_clusters,
            "expired_clusters": expired_clusters,
            "total_clusters": len(self.clusters),
            "large_clusters": large_clusters,
            "average_cluster_size": (np.mean(cluster_sizes) if cluster_sizes else 0),
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "average_coherence": avg_coherence,
            "tweet_embeddings_stored": len(self.tweet_embeddings),
            "birch_fitted": self.birch_fitted,
            "unique_birch_labels": len(self.birch_label_to_cluster_id),
        }

    def rebuild_index(self):
        """Rebuild BIRCH clustering from scratch (useful after many deletions)"""
        logger.info("Rebuilding BIRCH clustering...")

        if len(self.tweet_embeddings) == 0:
            logger.warning("No tweet embeddings to rebuild BIRCH clustering")
            return

        try:
            # Get all active tweet embeddings
            active_embeddings = []
            active_tweet_ids = []

            for cluster in self.clusters.values():
                if cluster.status == ClusterStatus.ACTIVE:
                    for tweet_id in cluster.tweet_ids:
                        if tweet_id in self.tweet_embeddings:
                            active_embeddings.append(self.tweet_embeddings[tweet_id])
                            active_tweet_ids.append(tweet_id)

            if len(active_embeddings) == 0:
                logger.warning("No active embeddings found for BIRCH rebuild")
                return

            # Create new BIRCH clusterer
            new_birch = Birch(
                n_clusters=None,
                threshold=max(0.2, 0.5 - len(active_embeddings) / 10000),
                branching_factor=50,
                compute_labels=True,
            )

            # Fit on active embeddings
            new_birch.fit(active_embeddings)

            # Update the clusterer
            self.birch_clusterer = new_birch
            self.birch_fitted = True

            logger.info(
                f"BIRCH clustering rebuilt with {len(active_embeddings)} active embeddings"
            )

        except Exception as e:
            logger.error(f"Error rebuilding BIRCH clustering: {e}")

    def set_all_tweets_reference(self, tweets: Dict[str, Tweet]):
        """Set reference to all tweets for quality maintenance operations"""
        self._all_tweets = tweets
