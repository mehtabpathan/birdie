#!/usr/bin/env python3
"""
🐦 Ultra-Simple Tweet Clustering System
Just the bare essentials - no complexity, just clustering that works!
Now with FAISS for fast similarity search!
"""

import json
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleTweet:
    """Just the basics we need"""

    def __init__(self, tweet_id: str, text: str, author: str = ""):
        self.id = tweet_id
        self.text = text
        self.author = author
        self.timestamp = datetime.now()


class SimpleCluster:
    """Minimal cluster representation"""

    def __init__(self, cluster_id: int):
        self.id = cluster_id
        self.tweets: List[SimpleTweet] = []
        self.centroid: Optional[np.ndarray] = None

    def add_tweet(self, tweet: SimpleTweet, embedding: np.ndarray):
        self.tweets.append(tweet)
        # Update centroid
        if self.centroid is None:
            self.centroid = embedding.copy()
        else:
            # Simple average
            self.centroid = (self.centroid * (len(self.tweets) - 1) + embedding) / len(
                self.tweets
            )

        # Normalize centroid for cosine similarity
        if self.centroid is not None:
            self.centroid = self.centroid / np.linalg.norm(self.centroid)

    @property
    def size(self) -> int:
        return len(self.tweets)


class SimpleClustering:
    """Dead simple clustering - just what you need, now with FAISS speed!"""

    def __init__(self):
        # Load embedding model (this is the only "complex" part)
        print("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        # Initialize FAISS index for fast similarity search
        print("Initializing FAISS index...")
        self.faiss_index = faiss.IndexFlatIP(
            self.embedding_dim
        )  # Inner Product for cosine similarity

        # Storage
        self.tweets: Dict[str, SimpleTweet] = {}
        self.clusters: Dict[int, SimpleCluster] = {}
        self.tweet_embeddings: Dict[str, np.ndarray] = {}

        # Mapping between FAISS index positions and cluster IDs
        self.faiss_id_to_cluster_id: List[int] = []  # faiss_index[i] -> cluster_id

        print("✅ Simple clustering system ready with FAISS acceleration!")

    def add_tweet(self, tweet_id: str, text: str, author: str = "") -> bool:
        """Add a tweet and cluster it"""
        try:
            # Create tweet
            tweet = SimpleTweet(tweet_id, text, author)
            self.tweets[tweet_id] = tweet

            # Get embedding and normalize for cosine similarity
            embedding = self.embedder.encode([text])[0]
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            self.tweet_embeddings[tweet_id] = embedding

            # Cluster it
            self._cluster_tweet(tweet, embedding)

            return True

        except Exception as e:
            print(f"Error adding tweet: {e}")
            return False

    def _cluster_tweet(self, tweet: SimpleTweet, embedding: np.ndarray):
        """Fast clustering logic using FAISS"""

        # If we have no clusters yet, create first one
        if not self.clusters:
            cluster = SimpleCluster(0)
            cluster.add_tweet(tweet, embedding)
            self.clusters[0] = cluster

            # Add to FAISS index
            self.faiss_index.add(cluster.centroid.reshape(1, -1))
            self.faiss_id_to_cluster_id.append(0)
            return

        # Use FAISS to find most similar cluster centroid
        similarities, faiss_indices = self.faiss_index.search(
            embedding.reshape(1, -1), k=1
        )

        best_similarity = similarities[0][
            0
        ]  # FAISS returns inner product (cosine similarity for normalized vectors)
        best_faiss_idx = faiss_indices[0][0]
        best_cluster_id = self.faiss_id_to_cluster_id[best_faiss_idx]

        # If similarity is good enough, add to existing cluster
        if best_similarity > 0.5:  # Simple threshold
            old_centroid = self.clusters[best_cluster_id].centroid.copy()
            self.clusters[best_cluster_id].add_tweet(tweet, embedding)

            # Update FAISS index with new centroid
            new_centroid = self.clusters[best_cluster_id].centroid
            self.faiss_index.remove_ids(np.array([best_faiss_idx], dtype=np.int64))
            self.faiss_index.add(new_centroid.reshape(1, -1))
            # Note: FAISS remove_ids changes indices, so we rebuild the mapping
            self._rebuild_faiss_mapping()

        else:
            # Create new cluster
            new_id = max(self.clusters.keys()) + 1
            cluster = SimpleCluster(new_id)
            cluster.add_tweet(tweet, embedding)
            self.clusters[new_id] = cluster

            # Add new centroid to FAISS index
            self.faiss_index.add(cluster.centroid.reshape(1, -1))
            self.faiss_id_to_cluster_id.append(new_id)

    def _rebuild_faiss_mapping(self):
        """Rebuild FAISS index and mapping after removals"""
        # Get all current centroids
        centroids = []
        cluster_ids = []

        for cluster_id, cluster in self.clusters.items():
            if cluster.centroid is not None:
                centroids.append(cluster.centroid)
                cluster_ids.append(cluster_id)

        if centroids:
            # Rebuild FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            centroids_array = np.array(centroids)
            self.faiss_index.add(centroids_array)
            self.faiss_id_to_cluster_id = cluster_ids

    def get_clusters(self) -> Dict[int, Dict]:
        """Get simple cluster info"""
        result = {}
        for cluster_id, cluster in self.clusters.items():
            result[cluster_id] = {
                "id": cluster_id,
                "size": cluster.size,
                "tweets": [
                    {"id": t.id, "text": t.text, "author": t.author}
                    for t in cluster.tweets
                ],
            }
        return result

    def get_stats(self) -> Dict:
        """Basic stats"""
        return {
            "total_tweets": len(self.tweets),
            "total_clusters": len(self.clusters),
            "largest_cluster": max((c.size for c in self.clusters.values()), default=0),
            "average_cluster_size": (
                np.mean([c.size for c in self.clusters.values()])
                if self.clusters
                else 0
            ),
            "faiss_index_size": self.faiss_index.ntotal,
        }

    def save(self, filename: str = "simple_clusters.pkl"):
        """Save everything including FAISS index"""
        data = {
            "tweets": self.tweets,
            "clusters": self.clusters,
            "embeddings": self.tweet_embeddings,
            "faiss_id_to_cluster_id": self.faiss_id_to_cluster_id,
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

        # Save FAISS index separately
        faiss_filename = filename.replace(".pkl", "_faiss.index")
        faiss.write_index(self.faiss_index, faiss_filename)

        print(f"💾 Saved to {filename} and {faiss_filename}")

    def load(self, filename: str = "simple_clusters.pkl"):
        """Load everything including FAISS index"""
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)

            self.tweets = data["tweets"]
            self.clusters = data["clusters"]
            self.tweet_embeddings = data["embeddings"]
            self.faiss_id_to_cluster_id = data.get("faiss_id_to_cluster_id", [])

            # Load FAISS index
            faiss_filename = filename.replace(".pkl", "_faiss.index")
            try:
                self.faiss_index = faiss.read_index(faiss_filename)
                print(f"📂 Loaded from {filename} and {faiss_filename}")
            except:
                print(
                    f"📂 Loaded from {filename} (FAISS index not found, rebuilding...)"
                )
                self._rebuild_faiss_mapping()

            return True
        except FileNotFoundError:
            print(f"File {filename} not found")
            return False


def demo():
    """Simple demo with speed test"""
    print("� Starting Simple Tweet Clustering Demo with FAISS")

    # Create clusterer
    clusterer = SimpleClustering()

    # Sample tweets
    sample_tweets = [
        ("1", "Machine learning is revolutionizing AI development", "user1"),
        ("2", "AI and ML are changing the world of technology", "user2"),
        ("3", "Love this sunny weather today! Perfect for a walk", "user3"),
        ("4", "Beautiful day outside, going for a hike", "user4"),
        ("5", "Deep learning models are getting more sophisticated", "user5"),
        ("6", "Neural networks are the future of AI", "user6"),
        ("7", "Rainy day, staying inside with a good book", "user7"),
        ("8", "Weather forecast shows storms coming", "user8"),
        ("9", "Python is great for machine learning projects", "user9"),
        ("10", "Coding in Python makes ML so much easier", "user10"),
        # Add more tweets to test FAISS performance
        ("11", "Artificial intelligence is transforming industries", "user11"),
        ("12", "Sunny skies and warm temperatures today", "user12"),
        ("13", "Programming with Python is so intuitive", "user13"),
        ("14", "Cloudy weather with chance of rain", "user14"),
        ("15", "Deep neural networks achieve amazing results", "user15"),
    ]

    # Add tweets
    print("\n📝 Adding tweets...")
    import time

    start_time = time.time()

    for tweet_id, text, author in sample_tweets:
        success = clusterer.add_tweet(tweet_id, text, author)
        if success:
            print(f"✅ Added: {text[:50]}...")

    end_time = time.time()
    print(f"⚡ Processing time: {end_time - start_time:.3f} seconds")

    # Show results
    print("\n📊 Clustering Results:")
    clusters = clusterer.get_clusters()

    for cluster_id, cluster_info in clusters.items():
        print(f"\n🎯 Cluster {cluster_id} ({cluster_info['size']} tweets):")
        for tweet in cluster_info["tweets"]:
            print(f"   • {tweet['text']}")

    # Show stats
    print("\n📈 Statistics:")
    stats = clusterer.get_stats()
    for key, value in stats.items():
        print(f"   • {key}: {value}")

    # Save
    clusterer.save()

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo()
