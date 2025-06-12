import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

from clustering_engine import IncrementalClusteringEngine
from models import Cluster, ClusterStatus, Tweet


class ClusterVisualizer:
    def __init__(
        self, clustering_engine: IncrementalClusteringEngine, tweets: Dict[str, Tweet]
    ):
        self.clustering_engine = clustering_engine
        self.tweets = tweets

        # Set up plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def create_comprehensive_dashboard(self, save_path: str = "cluster_dashboard.html"):
        """Create a comprehensive interactive dashboard"""
        print("🎨 Creating comprehensive clustering dashboard...")

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=3,
            subplot_titles=[
                "Cluster Size Distribution",
                "Topic Distribution",
                "Quality Metrics",
                "Cluster Timeline",
                "BIRCH Labels Distribution",
                "Coherence vs Size",
                "Tweet Embeddings (2D)",
                "Top Clusters by Size",
                "Processing Statistics",
            ],
            specs=[
                [{"type": "histogram"}, {"type": "pie"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}],
            ],
        )

        # Get data
        stats = self.clustering_engine.get_statistics()
        cluster_data = self._prepare_cluster_data()

        # 1. Cluster Size Distribution
        cluster_sizes = [c["size"] for c in cluster_data if c["status"] == "ACTIVE"]
        fig.add_trace(
            go.Histogram(x=cluster_sizes, nbinsx=20, name="Cluster Sizes"), row=1, col=1
        )

        # 2. Topic Distribution
        topic_counts = Counter(
            [c["topic_match"] for c in cluster_data if c["topic_match"]]
        )
        fig.add_trace(
            go.Pie(
                labels=list(topic_counts.keys()), values=list(topic_counts.values())
            ),
            row=1,
            col=2,
        )

        # 3. Quality Metrics
        quality_metrics = [
            "Average Coherence",
            "Large Clusters",
            "Active Clusters",
            "Tweet Coverage",
        ]
        quality_values = [
            stats["average_coherence"],
            stats["large_clusters"] / max(stats["active_clusters"], 1),
            stats["active_clusters"] / max(stats["total_clusters"], 1),
            stats["tweets_clustered"] / max(stats["tweets_processed"], 1),
        ]
        fig.add_trace(
            go.Bar(x=quality_metrics, y=quality_values, name="Quality"), row=1, col=3
        )

        # 4. Cluster Timeline
        timeline_data = self._get_cluster_timeline()
        if timeline_data:
            fig.add_trace(
                go.Scatter(
                    x=timeline_data["hours"],
                    y=timeline_data["cumulative_clusters"],
                    mode="lines+markers",
                    name="Clusters Created",
                ),
                row=2,
                col=1,
            )

        # 5. BIRCH Labels Distribution
        birch_labels = [
            c["birch_label"] for c in cluster_data if c["birch_label"] is not None
        ]
        if birch_labels:
            label_counts = Counter(birch_labels)
            top_labels = dict(
                sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            )
            fig.add_trace(
                go.Bar(
                    x=list(map(str, top_labels.keys())), y=list(top_labels.values())
                ),
                row=2,
                col=2,
            )

        # 6. Coherence vs Size
        coherences = [c["coherence"] for c in cluster_data if c["coherence"] > 0]
        sizes_for_coherence = [c["size"] for c in cluster_data if c["coherence"] > 0]
        if coherences and sizes_for_coherence:
            fig.add_trace(
                go.Scatter(
                    x=sizes_for_coherence,
                    y=coherences,
                    mode="markers",
                    name="Coherence vs Size",
                    text=[
                        f"Cluster {c['id'][:8]}"
                        for c in cluster_data
                        if c["coherence"] > 0
                    ],
                ),
                row=2,
                col=3,
            )

        # 7. Tweet Embeddings Visualization (2D projection)
        embedding_plot = self._create_embedding_visualization()
        if embedding_plot:
            fig.add_trace(embedding_plot, row=3, col=1)

        # 8. Top Clusters by Size
        top_clusters = sorted(cluster_data, key=lambda x: x["size"], reverse=True)[:10]
        fig.add_trace(
            go.Bar(
                x=[f"C{c['id'][:6]}" for c in top_clusters],
                y=[c["size"] for c in top_clusters],
                text=[c["topic_match"] or "No Topic" for c in top_clusters],
                name="Top Clusters",
            ),
            row=3,
            col=2,
        )

        # 9. Processing Statistics
        processing_stats = [
            "Tweets Processed",
            "Clusters Created",
            "Tweets Clustered",
            "Quality Checks",
        ]
        processing_values = [
            stats["tweets_processed"],
            stats["clusters_created"],
            stats["tweets_clustered"],
            stats["quality_checks_performed"],
        ]
        fig.add_trace(
            go.Bar(x=processing_stats, y=processing_values, name="Processing"),
            row=3,
            col=3,
        )

        # Update layout
        fig.update_layout(
            height=1200,
            title_text="🐦 BIRCH Tweet Clustering Dashboard",
            title_x=0.5,
            showlegend=False,
            template="plotly_white",
        )

        # Save dashboard
        pyo.plot(fig, filename=save_path, auto_open=False)
        print(f"✅ Dashboard saved to {save_path}")

        return fig

    def create_cluster_quality_report(
        self, save_path: str = "cluster_quality_report.html"
    ):
        """Create detailed cluster quality analysis"""
        print("📊 Creating cluster quality report...")

        cluster_data = self._prepare_cluster_data()

        # Create quality analysis plots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Coherence Distribution",
                "Size vs Topic Similarity",
                "Quality Score Distribution",
                "Cluster Health Matrix",
            ],
        )

        # 1. Coherence Distribution
        coherences = [c["coherence"] for c in cluster_data if c["coherence"] > 0]
        fig.add_trace(
            go.Histogram(x=coherences, nbinsx=20, name="Coherence"), row=1, col=1
        )

        # 2. Size vs Topic Similarity
        sizes = [c["size"] for c in cluster_data if c["topic_similarity"] > 0]
        similarities = [
            c["topic_similarity"] for c in cluster_data if c["topic_similarity"] > 0
        ]
        topics = [
            c["topic_match"] or "None"
            for c in cluster_data
            if c["topic_similarity"] > 0
        ]

        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=similarities,
                mode="markers",
                text=topics,
                name="Size vs Similarity",
                marker=dict(size=8, opacity=0.7),
            ),
            row=1,
            col=2,
        )

        # 3. Quality Score Distribution
        quality_scores = []
        for c in cluster_data:
            if c["status"] == "ACTIVE":
                # Calculate composite quality score
                coherence_score = c["coherence"] if c["coherence"] > 0 else 0
                size_score = min(c["size"] / 10, 1.0)  # Normalize size
                topic_score = c["topic_similarity"] if c["topic_similarity"] > 0 else 0

                quality = (coherence_score + size_score + topic_score) / 3
                quality_scores.append(quality)

        fig.add_trace(
            go.Histogram(x=quality_scores, nbinsx=15, name="Quality Score"),
            row=2,
            col=1,
        )

        # 4. Cluster Health Matrix
        health_matrix = self._create_health_matrix(cluster_data)
        fig.add_trace(
            go.Heatmap(
                z=health_matrix["values"],
                x=health_matrix["x_labels"],
                y=health_matrix["y_labels"],
                colorscale="RdYlGn",
                name="Health Matrix",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800,
            title_text="🔍 Cluster Quality Analysis Report",
            title_x=0.5,
            template="plotly_white",
        )

        pyo.plot(fig, filename=save_path, auto_open=False)
        print(f"✅ Quality report saved to {save_path}")

        return fig

    def create_topic_analysis(self, save_path: str = "topic_analysis.html"):
        """Create topic-focused analysis"""
        print("🎯 Creating topic analysis...")

        cluster_data = self._prepare_cluster_data()

        # Group by topics
        topic_analysis = defaultdict(list)
        for c in cluster_data:
            if c["topic_match"] and c["status"] == "ACTIVE":
                topic_analysis[c["topic_match"]].append(c)

        # Create topic comparison
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Clusters per Topic",
                "Average Similarity by Topic",
                "Topic Coverage Over Time",
                "Topic Quality Comparison",
            ],
        )

        # 1. Clusters per Topic
        topic_names = list(topic_analysis.keys())
        cluster_counts = [len(clusters) for clusters in topic_analysis.values()]

        fig.add_trace(
            go.Bar(x=topic_names, y=cluster_counts, name="Cluster Count"), row=1, col=1
        )

        # 2. Average Similarity by Topic
        avg_similarities = []
        for topic, clusters in topic_analysis.items():
            similarities = [
                c["topic_similarity"] for c in clusters if c["topic_similarity"] > 0
            ]
            avg_similarities.append(np.mean(similarities) if similarities else 0)

        fig.add_trace(
            go.Bar(x=topic_names, y=avg_similarities, name="Avg Similarity"),
            row=1,
            col=2,
        )

        # 3. Topic Coverage Over Time (simplified)
        timeline_data = self._get_topic_timeline()
        if timeline_data:
            for topic, data in timeline_data.items():
                fig.add_trace(
                    go.Scatter(
                        x=data["time"],
                        y=data["cumulative"],
                        mode="lines",
                        name=f"{topic} Coverage",
                    ),
                    row=2,
                    col=1,
                )

        # 4. Topic Quality Comparison
        topic_quality = {}
        for topic, clusters in topic_analysis.items():
            coherences = [c["coherence"] for c in clusters if c["coherence"] > 0]
            sizes = [c["size"] for c in clusters]

            avg_coherence = np.mean(coherences) if coherences else 0
            avg_size = np.mean(sizes) if sizes else 0

            # Composite quality score
            quality = (avg_coherence + min(avg_size / 5, 1.0)) / 2
            topic_quality[topic] = quality

        fig.add_trace(
            go.Bar(
                x=list(topic_quality.keys()),
                y=list(topic_quality.values()),
                name="Topic Quality",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800,
            title_text="🎯 Topic Analysis Dashboard",
            title_x=0.5,
            template="plotly_white",
        )

        pyo.plot(fig, filename=save_path, auto_open=False)
        print(f"✅ Topic analysis saved to {save_path}")

        return fig

    def create_static_summary_plots(self):
        """Create static matplotlib plots for quick overview"""
        print("📈 Creating static summary plots...")

        cluster_data = self._prepare_cluster_data()
        stats = self.clustering_engine.get_statistics()

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "BIRCH Clustering Analysis Summary", fontsize=16, fontweight="bold"
        )

        # 1. Cluster Size Distribution
        cluster_sizes = [c["size"] for c in cluster_data if c["status"] == "ACTIVE"]
        axes[0, 0].hist(
            cluster_sizes, bins=20, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].set_title("Cluster Size Distribution")
        axes[0, 0].set_xlabel("Cluster Size")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Topic Distribution
        topic_counts = Counter(
            [c["topic_match"] for c in cluster_data if c["topic_match"]]
        )
        if topic_counts:
            axes[0, 1].pie(
                topic_counts.values(), labels=topic_counts.keys(), autopct="%1.1f%%"
            )
            axes[0, 1].set_title("Topic Distribution")

        # 3. Coherence vs Size Scatter
        coherences = [c["coherence"] for c in cluster_data if c["coherence"] > 0]
        sizes_for_coherence = [c["size"] for c in cluster_data if c["coherence"] > 0]
        if coherences and sizes_for_coherence:
            scatter = axes[0, 2].scatter(
                sizes_for_coherence, coherences, alpha=0.6, c="coral"
            )
            axes[0, 2].set_title("Cluster Coherence vs Size")
            axes[0, 2].set_xlabel("Cluster Size")
            axes[0, 2].set_ylabel("Coherence Score")
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Processing Statistics
        processing_stats = [
            "Tweets\nProcessed",
            "Clusters\nCreated",
            "Tweets\nClustered",
            "Quality\nChecks",
        ]
        processing_values = [
            stats["tweets_processed"],
            stats["clusters_created"],
            stats["tweets_clustered"],
            stats["quality_checks_performed"],
        ]
        bars = axes[1, 0].bar(
            processing_stats,
            processing_values,
            color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
        )
        axes[1, 0].set_title("Processing Statistics")
        axes[1, 0].set_ylabel("Count")

        # Add value labels on bars
        for bar, value in zip(bars, processing_values):
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(processing_values) * 0.01,
                str(value),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 5. Quality Metrics
        quality_metrics = [
            "Avg\nCoherence",
            "Active\nClusters",
            "Large\nClusters",
            "Coverage\nRate",
        ]
        quality_values = [
            stats["average_coherence"],
            stats["active_clusters"],
            stats["large_clusters"],
            stats["tweets_clustered"] / max(stats["tweets_processed"], 1),
        ]

        # Normalize for better visualization
        normalized_values = []
        for i, (metric, value) in enumerate(zip(quality_metrics, quality_values)):
            if i == 0:  # Coherence (0-1)
                normalized_values.append(value)
            elif i == 3:  # Coverage rate (0-1)
                normalized_values.append(value)
            else:  # Counts - normalize by max
                normalized_values.append(
                    value / max(quality_values[1:3])
                    if max(quality_values[1:3]) > 0
                    else 0
                )

        bars = axes[1, 1].bar(
            quality_metrics,
            normalized_values,
            color=["#FFD93D", "#6BCF7F", "#4D96FF", "#FF6B9D"],
        )
        axes[1, 1].set_title("Quality Metrics (Normalized)")
        axes[1, 1].set_ylabel("Normalized Score")
        axes[1, 1].set_ylim(0, 1.1)

        # 6. Top Clusters
        top_clusters = sorted(cluster_data, key=lambda x: x["size"], reverse=True)[:8]
        cluster_names = [f"C{c['id'][:6]}" for c in top_clusters]
        cluster_sizes_top = [c["size"] for c in top_clusters]

        bars = axes[1, 2].bar(cluster_names, cluster_sizes_top, color="lightcoral")
        axes[1, 2].set_title("Top Clusters by Size")
        axes[1, 2].set_xlabel("Cluster ID")
        axes[1, 2].set_ylabel("Size")
        axes[1, 2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("clustering_summary.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("✅ Static plots saved as 'clustering_summary.png'")

    def _prepare_cluster_data(self) -> List[Dict]:
        """Prepare cluster data for visualization"""
        cluster_data = []

        for cluster_id, cluster in self.clustering_engine.clusters.items():
            # Calculate coherence
            coherence = self.clustering_engine._calculate_cluster_coherence(cluster_id)

            # Get BIRCH label
            birch_label = self.clustering_engine.cluster_id_to_birch_label.get(
                cluster_id
            )

            cluster_info = {
                "id": cluster_id,
                "size": cluster.size,
                "status": cluster.status.value,
                "topic_match": cluster.topic_match,
                "topic_similarity": cluster.topic_similarity or 0,
                "coherence": coherence,
                "created_at": cluster.created_at,
                "last_tweet_added": cluster.last_tweet_added,
                "birch_label": birch_label,
            }
            cluster_data.append(cluster_info)

        return cluster_data

    def _get_cluster_timeline(self) -> Optional[Dict]:
        """Get cluster creation timeline"""
        cluster_data = self._prepare_cluster_data()

        if not cluster_data:
            return None

        # Sort by creation time
        sorted_clusters = sorted(cluster_data, key=lambda x: x["created_at"])

        # Create timeline
        timeline = {"hours": [], "cumulative_clusters": []}

        start_time = sorted_clusters[0]["created_at"]
        cumulative = 0

        for cluster in sorted_clusters:
            hours_diff = (cluster["created_at"] - start_time).total_seconds() / 3600
            cumulative += 1

            timeline["hours"].append(hours_diff)
            timeline["cumulative_clusters"].append(cumulative)

        return timeline

    def _get_topic_timeline(self) -> Dict:
        """Get topic coverage timeline"""
        cluster_data = self._prepare_cluster_data()

        # Group by topic and sort by time
        topic_timelines = defaultdict(list)

        for cluster in cluster_data:
            if cluster["topic_match"] and cluster["status"] == "ACTIVE":
                topic_timelines[cluster["topic_match"]].append(cluster["created_at"])

        # Create cumulative timelines
        result = {}
        for topic, timestamps in topic_timelines.items():
            timestamps.sort()
            if timestamps:
                start_time = timestamps[0]
                timeline = {"time": [], "cumulative": []}

                for i, timestamp in enumerate(timestamps):
                    hours_diff = (timestamp - start_time).total_seconds() / 3600
                    timeline["time"].append(hours_diff)
                    timeline["cumulative"].append(i + 1)

                result[topic] = timeline

        return result

    def _create_embedding_visualization(self) -> Optional[go.Scatter]:
        """Create 2D visualization of tweet embeddings"""
        try:
            # Get embeddings and labels
            embeddings = []
            cluster_labels = []
            tweet_ids = []

            for tweet_id, embedding in list(
                self.clustering_engine.tweet_embeddings.items()
            )[
                :500
            ]:  # Limit for performance
                embeddings.append(embedding)

                # Find cluster for this tweet
                cluster_id = None
                for cid, cluster in self.clustering_engine.clusters.items():
                    if tweet_id in cluster.tweet_ids:
                        cluster_id = cid
                        break

                cluster_labels.append(cluster_id[:8] if cluster_id else "Unknown")
                tweet_ids.append(tweet_id)

            if len(embeddings) < 2:
                return None

            # Reduce dimensionality
            embeddings_array = np.array(embeddings)

            if embeddings_array.shape[1] > 50:
                # First reduce with PCA if very high dimensional
                pca = PCA(n_components=50)
                embeddings_reduced = pca.fit_transform(embeddings_array)
            else:
                embeddings_reduced = embeddings_array

            # Use t-SNE for final 2D projection
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(embeddings) // 4),
            )
            embeddings_2d = tsne.fit_transform(embeddings_reduced)

            return go.Scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                mode="markers",
                text=cluster_labels,
                name="Tweet Embeddings",
                marker=dict(size=4, opacity=0.6),
            )

        except Exception as e:
            print(f"Warning: Could not create embedding visualization: {e}")
            return None

    def _create_health_matrix(self, cluster_data: List[Dict]) -> Dict:
        """Create cluster health matrix"""
        # Define health categories
        size_bins = ["Small (1-2)", "Medium (3-10)", "Large (11+)"]
        coherence_bins = ["Low (<0.5)", "Medium (0.5-0.8)", "High (0.8+)"]

        # Initialize matrix
        matrix = np.zeros((len(coherence_bins), len(size_bins)))

        for cluster in cluster_data:
            if cluster["status"] == "ACTIVE":
                # Determine size bin
                size = cluster["size"]
                if size <= 2:
                    size_idx = 0
                elif size <= 10:
                    size_idx = 1
                else:
                    size_idx = 2

                # Determine coherence bin
                coherence = cluster["coherence"]
                if coherence < 0.5:
                    coherence_idx = 0
                elif coherence < 0.8:
                    coherence_idx = 1
                else:
                    coherence_idx = 2

                matrix[coherence_idx, size_idx] += 1

        return {
            "values": matrix.tolist(),
            "x_labels": size_bins,
            "y_labels": coherence_bins,
        }

    def print_cluster_summary(self):
        """Print a text summary of clustering results"""
        stats = self.clustering_engine.get_statistics()
        cluster_data = self._prepare_cluster_data()

        print("\n" + "=" * 60)
        print("🐦 BIRCH CLUSTERING SUMMARY REPORT")
        print("=" * 60)

        print(f"\n📊 PROCESSING STATISTICS:")
        print(f"   • Total tweets processed: {stats['tweets_processed']:,}")
        print(f"   • Tweets successfully clustered: {stats['tweets_clustered']:,}")
        print(
            f"   • Clustering success rate: {stats['tweets_clustered']/max(stats['tweets_processed'], 1)*100:.1f}%"
        )

        print(f"\n🎯 CLUSTER STATISTICS:")
        print(f"   • Total clusters created: {stats['clusters_created']:,}")
        print(f"   • Currently active clusters: {stats['active_clusters']:,}")
        print(f"   • Large clusters (>50 tweets): {stats['large_clusters']:,}")
        print(f"   • Average cluster size: {stats['average_cluster_size']:.2f}")
        print(f"   • Maximum cluster size: {stats['max_cluster_size']:,}")

        print(f"\n🔍 QUALITY METRICS:")
        print(f"   • Average cluster coherence: {stats['average_coherence']:.3f}")
        print(f"   • Quality checks performed: {stats['quality_checks_performed']:,}")
        print(f"   • BIRCH model refits: {stats['birch_refits']:,}")

        # Topic analysis
        topic_counts = Counter(
            [c["topic_match"] for c in cluster_data if c["topic_match"]]
        )
        if topic_counts:
            print(f"\n🎯 TOPIC DISTRIBUTION:")
            for topic, count in topic_counts.most_common():
                percentage = (
                    count / len([c for c in cluster_data if c["topic_match"]]) * 100
                )
                print(f"   • {topic}: {count} clusters ({percentage:.1f}%)")

        # Top clusters
        top_clusters = sorted(cluster_data, key=lambda x: x["size"], reverse=True)[:5]
        if top_clusters:
            print(f"\n🏆 TOP 5 CLUSTERS BY SIZE:")
            for i, cluster in enumerate(top_clusters, 1):
                topic = cluster["topic_match"] or "No Topic"
                print(f"   {i}. Cluster {cluster['id'][:8]}: {cluster['size']} tweets")
                print(
                    f"      Topic: {topic} (similarity: {cluster['topic_similarity']:.3f})"
                )
                print(f"      Coherence: {cluster['coherence']:.3f}")

        print("\n" + "=" * 60)
