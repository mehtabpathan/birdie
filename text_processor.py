import re
import string
from typing import List, Set

import nltk
from textblob import TextBlob

from models import Tweet


class TextProcessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        from nltk.corpus import stopwords

        self.stop_words = set(stopwords.words("english"))

        # Common social media words to remove
        self.social_media_words = {
            "rt",
            "retweet",
            "via",
            "follow",
            "following",
            "followers",
            "like",
            "likes",
            "share",
            "shares",
            "comment",
            "comments",
            "dm",
            "pm",
            "tweet",
            "tweets",
            "twitter",
            "instagram",
            "facebook",
            "linkedin",
            "tiktok",
            "youtube",
            "snapchat",
        }

        # Combine stop words
        self.all_stop_words = self.stop_words.union(self.social_media_words)

    def clean_tweet_text(self, text: str) -> str:
        """Clean and preprocess tweet text"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )
        text = re.sub(
            r"www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove mentions (@username)
        text = re.sub(r"@[A-Za-z0-9_]+", "", text)

        # Remove hashtags but keep the text (remove # symbol)
        text = re.sub(r"#([A-Za-z0-9_]+)", r"\1", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove punctuation except for emoticons
        # Keep basic emoticons like :) :( :D etc.
        emoticon_pattern = r"[:\-=][)(\[\]{}|\\\/DPp]|[)(\[\]{}|\\\/DPp][:=\-]"
        emoticons = re.findall(emoticon_pattern, text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Add back emoticons
        for emoticon in emoticons:
            text += f" {emoticon}"

        # Remove numbers (optional - might want to keep some)
        text = re.sub(r"\d+", "", text)

        # Remove extra whitespace again
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def remove_stop_words(self, text: str) -> str:
        """Remove stop words from text"""
        words = text.split()
        filtered_words = [
            word for word in words if word.lower() not in self.all_stop_words
        ]
        return " ".join(filtered_words)

    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from text"""
        # Clean the text first
        cleaned_text = self.clean_tweet_text(text)

        # Remove stop words
        no_stop_words = self.remove_stop_words(cleaned_text)

        # Split into words and filter by length
        words = [word for word in no_stop_words.split() if len(word) >= min_length]

        return words

    def get_sentiment(self, text: str) -> dict:
        """Get sentiment analysis of the text"""
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment

            # Classify sentiment
            if sentiment.polarity > 0.1:
                sentiment_label = "positive"
            elif sentiment.polarity < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            return {
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity,
                "label": sentiment_label,
            }
        except Exception as e:
            return {"polarity": 0.0, "subjectivity": 0.0, "label": "neutral"}

    def process_tweet(self, tweet: Tweet) -> Tweet:
        """Process a single tweet - clean text and extract features"""
        # Clean the text
        cleaned_text = self.clean_tweet_text(tweet.text)

        # Remove stop words
        cleaned_text = self.remove_stop_words(cleaned_text)

        # Update the tweet
        tweet.cleaned_text = cleaned_text

        return tweet

    def process_tweets_batch(self, tweets: List[Tweet]) -> List[Tweet]:
        """Process a batch of tweets"""
        processed_tweets = []

        for tweet in tweets:
            try:
                processed_tweet = self.process_tweet(tweet)
                processed_tweets.append(processed_tweet)
            except Exception as e:
                print(f"Error processing tweet {tweet.id}: {e}")
                # Keep original tweet if processing fails
                processed_tweets.append(tweet)

        return processed_tweets

    def is_meaningful_text(self, text: str, min_words: int = 2) -> bool:
        """Check if text has meaningful content for clustering"""
        if not text or not text.strip():
            return False

        # Clean and process the text
        cleaned = self.clean_tweet_text(text)
        no_stop_words = self.remove_stop_words(cleaned)

        # Check if we have enough meaningful words
        words = [word for word in no_stop_words.split() if len(word) >= 3]

        return len(words) >= min_words

    def extract_entities(self, text: str) -> dict:
        """Extract named entities from text (simple version)"""
        # This is a simplified version - in production you might use spaCy or similar
        entities = {"companies": [], "locations": [], "people": [], "technologies": []}

        # Simple keyword-based entity extraction
        company_keywords = [
            "apple",
            "google",
            "microsoft",
            "amazon",
            "tesla",
            "meta",
            "netflix",
            "nvidia",
        ]
        tech_keywords = [
            "ai",
            "blockchain",
            "bitcoin",
            "ethereum",
            "machine learning",
            "deep learning",
        ]

        text_lower = text.lower()

        for company in company_keywords:
            if company in text_lower:
                entities["companies"].append(company.title())

        for tech in tech_keywords:
            if tech in text_lower:
                entities["technologies"].append(tech.title())

        return entities
