import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    # Twitter API Configuration
    TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    TWITTER_API_KEY: str = os.getenv("TWITTER_API_KEY", "")
    TWITTER_API_SECRET: str = os.getenv("TWITTER_API_SECRET", "")
    TWITTER_ACCESS_TOKEN: str = os.getenv("TWITTER_ACCESS_TOKEN", "")
    TWITTER_ACCESS_TOKEN_SECRET: str = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")

    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # Slack Configuration
    SLACK_BOT_TOKEN: str = os.getenv("SLACK_BOT_TOKEN", "")
    SLACK_CHANNEL: str = os.getenv("SLACK_CHANNEL", "#alerts")

    # Clustering Configuration
    MIN_CLUSTER_SIZE: int = int(os.getenv("MIN_CLUSTER_SIZE", "5"))
    CLUSTER_EXPIRY_HOURS: int = int(os.getenv("CLUSTER_EXPIRY_HOURS", "48"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Processing Configuration
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
    PROCESSING_INTERVAL: int = int(os.getenv("PROCESSING_INTERVAL", "30"))  # seconds

    # Topics of Interest
    TOPICS_OF_INTEREST: List[Dict[str, str]] = [
        {
            "title": "Technology Trends",
            "description": "Discussions about AI, machine learning, blockchain, and emerging technologies",
        },
        {
            "title": "Market Analysis",
            "description": "Stock market trends, cryptocurrency, financial news and analysis",
        },
        {
            "title": "Breaking News",
            "description": "Major news events, political developments, and current affairs",
        },
        {
            "title": "Product Launches",
            "description": "New product announcements, startup launches, and business developments",
        },
    ]

    class Config:
        env_file = ".env"


config = Config()
