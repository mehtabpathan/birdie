import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TweetStatus(str, Enum):
    ACTIVE = "active"
    CLUSTERED = "clustered"
    EXPIRED = "expired"


class ClusterStatus(str, Enum):
    ACTIVE = "active"
    ALERTED = "alerted"
    EXPIRED = "expired"


class Tweet(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tweet_id: str  # Original Twitter ID
    text: str
    cleaned_text: str = ""
    author_id: str
    author_username: str
    created_at: datetime
    retweet_count: int = 0
    like_count: int = 0
    reply_count: int = 0
    quote_count: int = 0
    hashtags: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    status: TweetStatus = TweetStatus.ACTIVE
    cluster_id: Optional[str] = None
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TopicOfInterest(BaseModel):
    title: str
    description: str
    keywords: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None


class Cluster(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tweet_ids: List[str] = Field(default_factory=list)
    centroid: Optional[List[float]] = None
    topic_match: Optional[str] = None  # Matched topic title
    topic_similarity: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_tweet_added: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status: ClusterStatus = ClusterStatus.ACTIVE
    alert_sent: bool = False
    size: int = 0
    summary: str = ""
    representative_tweets: List[str] = Field(default_factory=list)  # Top 3 tweet IDs

    def add_tweet(self, tweet_id: str):
        if tweet_id not in self.tweet_ids:
            self.tweet_ids.append(tweet_id)
            self.size = len(self.tweet_ids)
            self.last_tweet_added = datetime.now(timezone.utc)
            self.last_updated = datetime.now(timezone.utc)

    def is_expired(self, expiry_hours: int = 48) -> bool:
        time_diff = datetime.now(timezone.utc) - self.last_tweet_added
        return time_diff.total_seconds() > (expiry_hours * 3600)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ClusteringModel(BaseModel):
    model_name: str
    embedding_dimension: int
    similarity_threshold: float
    min_cluster_size: int
    total_tweets_processed: int = 0
    total_clusters_created: int = 0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AlertRule(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    min_cluster_size: int
    topic_similarity_threshold: float
    enabled: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProcessingStats(BaseModel):
    tweets_processed: int = 0
    tweets_cleaned: int = 0
    clusters_created: int = 0
    clusters_expired: int = 0
    alerts_sent: int = 0
    last_processing_time: Optional[datetime] = None
    processing_errors: int = 0

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
