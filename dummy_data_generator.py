import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from models import Tweet


class DummyTweetGenerator:
    def __init__(self):
        self.topics_templates = {
            "Technology Trends": {
                "templates": [
                    "Just tried the new {tech} feature and it's {adjective}! {hashtag}",
                    "Breaking: {company} announces breakthrough in {tech} technology",
                    "The future of {tech} is here! {opinion} #innovation",
                    "AI is revolutionizing {industry}. What are your thoughts?",
                    "New study shows {tech} adoption increased by {percentage}% this year",
                    "Machine learning models are getting {adjective} every day",
                    "Blockchain technology could transform {industry} forever",
                    "Just attended a conference on {tech} - mind blown! 🤯",
                    "The ethical implications of {tech} need more discussion",
                    "Quantum computing breakthrough: {achievement}",
                ],
                "tech": [
                    "AI",
                    "machine learning",
                    "blockchain",
                    "quantum computing",
                    "IoT",
                    "5G",
                    "AR/VR",
                    "robotics",
                    "neural networks",
                    "deep learning",
                ],
                "company": [
                    "Google",
                    "Microsoft",
                    "Apple",
                    "Tesla",
                    "OpenAI",
                    "Meta",
                    "Amazon",
                    "IBM",
                    "NVIDIA",
                    "Intel",
                ],
                "adjective": [
                    "amazing",
                    "revolutionary",
                    "game-changing",
                    "impressive",
                    "concerning",
                    "promising",
                    "sophisticated",
                    "advanced",
                ],
                "industry": [
                    "healthcare",
                    "finance",
                    "education",
                    "transportation",
                    "manufacturing",
                    "retail",
                    "agriculture",
                    "entertainment",
                ],
                "hashtag": [
                    "#AI",
                    "#MachineLearning",
                    "#Blockchain",
                    "#Innovation",
                    "#TechTrends",
                    "#FutureTech",
                    "#DigitalTransformation",
                ],
                "opinion": [
                    "This changes everything!",
                    "Still skeptical about this",
                    "Exciting times ahead",
                    "We need to be careful",
                ],
                "percentage": ["25", "40", "60", "75", "100", "150", "200"],
                "achievement": [
                    "new qubit record",
                    "error correction breakthrough",
                    "faster processing speeds",
                    "improved stability",
                ],
            },
            "Market Analysis": {
                "templates": [
                    "{stock} is {direction} {percentage}% today. {opinion} #stocks",
                    "Crypto market update: {crypto} hits {price_action} #cryptocurrency",
                    "Federal Reserve announces {fed_action}. Markets react {reaction}",
                    "Breaking: {company} reports {earnings_result} earnings",
                    "Oil prices {direction} amid {geopolitical_event}",
                    "Gold reaches {price_level} as investors seek {investment_reason}",
                    "Tech stocks {performance} after {market_event}",
                    "Inflation data shows {inflation_trend}. What does this mean for {asset}?",
                    "Market volatility increases due to {uncertainty_factor}",
                    "Analysts predict {prediction} for {timeframe}",
                ],
                "stock": [
                    "AAPL",
                    "GOOGL",
                    "TSLA",
                    "MSFT",
                    "AMZN",
                    "NVDA",
                    "META",
                    "NFLX",
                    "SPY",
                    "QQQ",
                ],
                "direction": [
                    "up",
                    "down",
                    "surging",
                    "plummeting",
                    "climbing",
                    "falling",
                ],
                "percentage": ["2", "5", "8", "12", "15", "20", "25"],
                "crypto": [
                    "Bitcoin",
                    "Ethereum",
                    "Solana",
                    "Cardano",
                    "Polygon",
                    "Chainlink",
                    "Dogecoin",
                ],
                "price_action": [
                    "new highs",
                    "support levels",
                    "resistance",
                    "all-time high",
                    "yearly low",
                ],
                "fed_action": [
                    "rate hike",
                    "rate cut",
                    "policy change",
                    "quantitative easing",
                ],
                "reaction": [
                    "positively",
                    "negatively",
                    "with volatility",
                    "cautiously",
                ],
                "company": [
                    "Apple",
                    "Google",
                    "Tesla",
                    "Microsoft",
                    "Amazon",
                    "Netflix",
                    "Meta",
                ],
                "earnings_result": [
                    "better than expected",
                    "disappointing",
                    "record-breaking",
                    "mixed",
                ],
                "geopolitical_event": [
                    "trade tensions",
                    "supply chain issues",
                    "political uncertainty",
                    "sanctions",
                ],
                "price_level": [
                    "new highs",
                    "$2000",
                    "resistance levels",
                    "key support",
                ],
                "investment_reason": [
                    "safe haven",
                    "hedge against inflation",
                    "portfolio diversification",
                ],
                "performance": [
                    "rally",
                    "decline",
                    "show mixed results",
                    "outperform market",
                ],
                "market_event": [
                    "earnings season",
                    "Fed meeting",
                    "economic data release",
                ],
                "inflation_trend": [
                    "rising concerns",
                    "cooling down",
                    "above expectations",
                    "stabilizing",
                ],
                "asset": ["stocks", "bonds", "commodities", "real estate"],
                "uncertainty_factor": [
                    "geopolitical tensions",
                    "economic data",
                    "policy changes",
                ],
                "prediction": [
                    "bullish outlook",
                    "bearish sentiment",
                    "sideways movement",
                    "increased volatility",
                ],
                "timeframe": ["Q4", "next year", "the coming months", "2024"],
                "opinion": [
                    "Time to buy the dip?",
                    "Profit taking time",
                    "Hold steady",
                    "Risky territory",
                ],
            },
            "Breaking News": {
                "templates": [
                    "BREAKING: {news_event} in {location}. {impact_statement}",
                    "UPDATE: {political_figure} announces {policy_change}",
                    "Emergency: {natural_disaster} hits {location}, {casualty_info}",
                    "URGENT: {international_event} sparks {global_reaction}",
                    "Live updates: {ongoing_event} continues to develop",
                    "Just in: {celebrity} {celebrity_action} #BreakingNews",
                    "Weather alert: {weather_event} expected in {region}",
                    "Sports: {team} {sports_result} in {tournament}",
                    "Health alert: {health_event} reported in {location}",
                    "Technology: Major {tech_incident} affects {affected_service}",
                ],
                "news_event": [
                    "major accident",
                    "political summit",
                    "protest",
                    "explosion",
                    "fire",
                    "rescue operation",
                ],
                "location": [
                    "New York",
                    "California",
                    "Texas",
                    "London",
                    "Tokyo",
                    "Berlin",
                    "Sydney",
                    "Toronto",
                ],
                "impact_statement": [
                    "Authorities investigating",
                    "No injuries reported",
                    "Traffic disrupted",
                    "Evacuations underway",
                ],
                "political_figure": [
                    "President",
                    "Prime Minister",
                    "Senator",
                    "Governor",
                    "Mayor",
                ],
                "policy_change": [
                    "new legislation",
                    "executive order",
                    "budget proposal",
                    "regulatory change",
                ],
                "natural_disaster": [
                    "earthquake",
                    "hurricane",
                    "wildfire",
                    "flood",
                    "tornado",
                    "blizzard",
                ],
                "casualty_info": [
                    "no casualties reported",
                    "rescue operations ongoing",
                    "emergency services responding",
                ],
                "international_event": [
                    "trade agreement",
                    "diplomatic meeting",
                    "border dispute",
                    "peace talks",
                ],
                "global_reaction": [
                    "international concern",
                    "market volatility",
                    "diplomatic tensions",
                ],
                "ongoing_event": [
                    "election coverage",
                    "court proceedings",
                    "investigation",
                    "negotiations",
                ],
                "celebrity": [
                    "Taylor Swift",
                    "Elon Musk",
                    "Oprah",
                    "The Rock",
                    "Beyoncé",
                    "Tom Hanks",
                ],
                "celebrity_action": [
                    "makes surprise announcement",
                    "donates to charity",
                    "launches new project",
                ],
                "weather_event": [
                    "severe storm",
                    "heat wave",
                    "cold front",
                    "heavy snow",
                    "flooding",
                ],
                "region": [
                    "Northeast",
                    "Southwest",
                    "Midwest",
                    "Pacific Coast",
                    "Southeast",
                ],
                "team": [
                    "Lakers",
                    "Yankees",
                    "Patriots",
                    "Warriors",
                    "Cowboys",
                    "Celtics",
                ],
                "sports_result": [
                    "wins championship",
                    "trades star player",
                    "upsets favorite",
                    "breaks record",
                ],
                "tournament": [
                    "playoffs",
                    "World Series",
                    "Super Bowl",
                    "NBA Finals",
                    "World Cup",
                ],
                "health_event": [
                    "outbreak",
                    "vaccine breakthrough",
                    "medical discovery",
                    "health advisory",
                ],
                "tech_incident": [
                    "data breach",
                    "system outage",
                    "cyber attack",
                    "software bug",
                ],
                "affected_service": [
                    "social media platforms",
                    "banking systems",
                    "cloud services",
                    "mobile networks",
                ],
            },
            "Product Launches": {
                "templates": [
                    "Excited to announce the launch of {product}! {feature_highlight} #ProductLaunch",
                    "{company} just dropped {product} and it's {reaction}! 🚀",
                    "First look at {product}: {review_snippet}",
                    "Pre-orders for {product} start {timeframe}. Who's getting one?",
                    "Behind the scenes: How we built {product} in {duration}",
                    "{product} is now available in {market}! {availability_info}",
                    "User review: {product} {user_experience} #ProductReview",
                    "Comparison: {product} vs {competitor} - which is better?",
                    "Limited edition {product} drops {timeframe}. Only {quantity} available!",
                    "The future of {category} is here with {product}",
                ],
                "product": [
                    "iPhone 15",
                    "Tesla Model Y",
                    "ChatGPT Plus",
                    "MacBook Pro",
                    "Galaxy S24",
                    "AirPods Pro",
                    "iPad Air",
                    "Surface Laptop",
                    "Pixel 8",
                    "Apple Watch",
                ],
                "feature_highlight": [
                    "Revolutionary camera system",
                    "All-day battery life",
                    "Lightning-fast performance",
                    "Stunning display quality",
                ],
                "company": [
                    "Apple",
                    "Tesla",
                    "Google",
                    "Microsoft",
                    "Samsung",
                    "Sony",
                    "Amazon",
                    "Meta",
                    "OpenAI",
                ],
                "reaction": [
                    "incredible",
                    "disappointing",
                    "overpriced",
                    "worth the wait",
                    "game-changing",
                    "underwhelming",
                ],
                "review_snippet": [
                    "Impressive build quality",
                    "Battery life could be better",
                    "Best in class performance",
                    "Great value for money",
                ],
                "timeframe": [
                    "tomorrow",
                    "next week",
                    "this Friday",
                    "December 1st",
                    "early 2024",
                ],
                "duration": ["6 months", "2 years", "18 months", "3 years"],
                "market": [
                    "Europe",
                    "Asia",
                    "North America",
                    "global markets",
                    "select countries",
                ],
                "availability_info": [
                    "Limited stock",
                    "Free shipping",
                    "Special launch price",
                    "Bundle deals available",
                ],
                "user_experience": [
                    "exceeded expectations",
                    "has some issues",
                    "perfect for my needs",
                    "not worth the hype",
                ],
                "competitor": [
                    "Samsung Galaxy",
                    "Google Pixel",
                    "OnePlus",
                    "Xiaomi",
                    "Huawei",
                ],
                "quantity": ["1000", "500", "2500", "100", "5000"],
                "category": [
                    "smartphones",
                    "laptops",
                    "wearables",
                    "smart home",
                    "gaming",
                    "productivity",
                ],
            },
            "General": {
                "templates": [
                    "Just had the best {food} at {location}! Highly recommend 👌",
                    "Monday motivation: {motivational_quote}",
                    "Can't believe it's already {time_reference}! Time flies ⏰",
                    "Watching {show} and it's {opinion}! Anyone else watching?",
                    "Travel tip: {travel_advice} #Travel",
                    "Life hack: {life_hack} - you're welcome! 💡",
                    "Grateful for {gratitude_item} today 🙏",
                    "Question for my followers: {question}",
                    "Throwback to {memory} - good times! #ThrowbackThursday",
                    "Currently reading {book} and {reading_opinion}",
                ],
                "food": [
                    "pizza",
                    "sushi",
                    "tacos",
                    "burger",
                    "pasta",
                    "ramen",
                    "coffee",
                    "ice cream",
                ],
                "location": [
                    "downtown",
                    "the new place",
                    "my favorite spot",
                    "this hidden gem",
                ],
                "motivational_quote": [
                    "Dream big, work hard",
                    "Success is a journey",
                    "Believe in yourself",
                    "Make it happen",
                ],
                "time_reference": [
                    "Friday",
                    "the weekend",
                    "December",
                    "2024",
                    "the holidays",
                ],
                "show": [
                    "Stranger Things",
                    "The Office",
                    "Game of Thrones",
                    "Breaking Bad",
                    "Friends",
                ],
                "opinion": [
                    "amazing",
                    "overrated",
                    "binge-worthy",
                    "confusing",
                    "hilarious",
                ],
                "travel_advice": [
                    "Pack light",
                    "Book early",
                    "Try local food",
                    "Learn basic phrases",
                ],
                "life_hack": [
                    "Use keyboard shortcuts",
                    "Meal prep on Sundays",
                    "Set phone to silent",
                    "Wake up early",
                ],
                "gratitude_item": [
                    "family",
                    "friends",
                    "health",
                    "opportunities",
                    "coffee",
                    "weekends",
                ],
                "question": [
                    "What's your favorite movie?",
                    "Coffee or tea?",
                    "Best travel destination?",
                    "Favorite book?",
                ],
                "memory": [
                    "last summer",
                    "college days",
                    "first job",
                    "childhood",
                    "last vacation",
                ],
                "book": ["Atomic Habits", "The Alchemist", "1984", "Sapiens", "Dune"],
                "reading_opinion": [
                    "loving it",
                    "can't put it down",
                    "it's okay",
                    "highly recommend it",
                ],
            },
        }

        self.usernames = [
            "techguru2024",
            "marketwatcher",
            "newsbreaker",
            "productfan",
            "cryptotrader",
            "aiexplorer",
            "startuplife",
            "innovator",
            "trendsetter",
            "analyst",
            "developer",
            "entrepreneur",
            "investor",
            "journalist",
            "blogger",
            "researcher",
            "consultant",
            "executive",
            "founder",
            "creator",
            "influencer",
            "expert",
            "specialist",
            "professional",
            "enthusiast",
        ]

        self.hashtags_by_topic = {
            "Technology Trends": [
                "#AI",
                "#MachineLearning",
                "#Blockchain",
                "#Innovation",
                "#TechTrends",
                "#FutureTech",
                "#DigitalTransformation",
                "#Automation",
                "#IoT",
                "#5G",
            ],
            "Market Analysis": [
                "#Stocks",
                "#Crypto",
                "#Trading",
                "#Investment",
                "#Finance",
                "#Markets",
                "#Economy",
                "#Bitcoin",
                "#Ethereum",
                "#Portfolio",
            ],
            "Breaking News": [
                "#BreakingNews",
                "#News",
                "#Update",
                "#Alert",
                "#Live",
                "#Urgent",
                "#Developing",
                "#Politics",
                "#World",
                "#Current",
            ],
            "Product Launches": [
                "#ProductLaunch",
                "#NewProduct",
                "#Innovation",
                "#Technology",
                "#Startup",
                "#Launch",
                "#Release",
                "#Announcement",
                "#Review",
                "#FirstLook",
            ],
            "General": [
                "#Life",
                "#Motivation",
                "#Inspiration",
                "#Thoughts",
                "#Daily",
                "#Personal",
                "#Lifestyle",
                "#Opinion",
                "#Question",
                "#Share",
            ],
        }

    def generate_tweet_text(self, topic: str) -> tuple[str, List[str]]:
        """Generate tweet text and extract hashtags"""
        if topic not in self.topics_templates:
            topic = "General"

        template_data = self.topics_templates[topic]
        template = random.choice(template_data["templates"])

        # Replace placeholders with random values
        for key, values in template_data.items():
            if key != "templates":
                placeholder = "{" + key + "}"
                if placeholder in template:
                    template = template.replace(placeholder, random.choice(values))

        # Add random hashtags
        hashtags = random.sample(self.hashtags_by_topic[topic], random.randint(1, 3))
        if not any(tag in template for tag in hashtags):
            template += " " + " ".join(hashtags)

        # Extract hashtags from the final text
        words = template.split()
        extracted_hashtags = [word[1:] for word in words if word.startswith("#")]

        return template, extracted_hashtags

    def generate_engagement_metrics(self, topic: str, hours_old: int) -> Dict[str, int]:
        """Generate realistic engagement metrics based on topic and age"""
        base_multipliers = {
            "Breaking News": 3.0,
            "Technology Trends": 2.0,
            "Market Analysis": 1.8,
            "Product Launches": 1.5,
            "General": 1.0,
        }

        multiplier = base_multipliers.get(topic, 1.0)

        # Older tweets generally have more engagement
        age_factor = min(1 + (hours_old / 24), 3.0)

        base_likes = random.randint(1, 100)
        base_retweets = random.randint(0, base_likes // 3)
        base_replies = random.randint(0, base_likes // 5)
        base_quotes = random.randint(0, base_retweets // 2)

        return {
            "like_count": int(base_likes * multiplier * age_factor),
            "retweet_count": int(base_retweets * multiplier * age_factor),
            "reply_count": int(base_replies * multiplier * age_factor),
            "quote_count": int(base_quotes * multiplier * age_factor),
        }

    def generate_dummy_tweets(self, count: int = 50000) -> List[Tweet]:
        """Generate a list of dummy tweets"""
        tweets = []
        topics = list(self.topics_templates.keys())

        # Weight topics (Breaking News and Tech Trends more common)
        topic_weights = [0.25, 0.25, 0.2, 0.15, 0.15]  # Corresponds to topics order

        print(f"Generating {count} dummy tweets...")

        for i in range(count):
            if i % 5000 == 0:
                print(f"Generated {i} tweets...")

            # Select topic based on weights
            topic = random.choices(topics, weights=topic_weights)[0]

            # Generate timestamp (spread over last 7 days)
            hours_ago = random.randint(0, 168)  # 7 days * 24 hours
            created_at = datetime.now(timezone.utc) - timedelta(hours=hours_ago)

            # Generate tweet text and hashtags
            text, hashtags = self.generate_tweet_text(topic)

            # Generate engagement metrics
            engagement = self.generate_engagement_metrics(topic, hours_ago)

            # Create tweet
            tweet = Tweet(
                tweet_id=f"tweet_{i+1:06d}",
                text=text,
                author_id=f"user_{random.randint(1, 10000):05d}",
                author_username=random.choice(self.usernames)
                + str(random.randint(1, 999)),
                created_at=created_at,
                hashtags=hashtags,
                mentions=[],  # Keep empty for simplicity
                urls=[],  # Keep empty for simplicity
                **engagement,
            )

            tweets.append(tweet)

        print(f"Successfully generated {count} tweets!")
        return tweets

    def save_tweets_to_json(
        self, tweets: List[Tweet], filename: str = "dummy_tweets.json"
    ):
        """Save tweets to JSON file"""
        tweets_data = [tweet.dict() for tweet in tweets]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(tweets_data, f, indent=2, default=str, ensure_ascii=False)

        print(f"Tweets saved to {filename}")

    def load_tweets_from_json(self, filename: str = "dummy_tweets.json") -> List[Tweet]:
        """Load tweets from JSON file"""
        with open(filename, "r", encoding="utf-8") as f:
            tweets_data = json.load(f)

        tweets = [Tweet(**tweet_data) for tweet_data in tweets_data]
        print(f"Loaded {len(tweets)} tweets from {filename}")
        return tweets


def main():
    """Generate and save dummy tweets"""
    generator = DummyTweetGenerator()

    # Generate 50,000 tweets
    tweets = generator.generate_dummy_tweets(50000)

    # Save to JSON file
    generator.save_tweets_to_json(tweets)

    # Print some statistics
    topics_count = {}
    for tweet in tweets:
        # Determine topic based on hashtags (simple heuristic)
        topic = "General"
        for hashtag in tweet.hashtags:
            if hashtag.lower() in ["ai", "machinelearning", "blockchain", "innovation"]:
                topic = "Technology Trends"
                break
            elif hashtag.lower() in ["stocks", "crypto", "trading", "bitcoin"]:
                topic = "Market Analysis"
                break
            elif hashtag.lower() in ["breakingnews", "news", "urgent"]:
                topic = "Breaking News"
                break
            elif hashtag.lower() in ["productlaunch", "newproduct", "launch"]:
                topic = "Product Launches"
                break

        topics_count[topic] = topics_count.get(topic, 0) + 1

    print("\nTweet distribution by topic:")
    for topic, count in topics_count.items():
        print(f"{topic}: {count} tweets ({count/len(tweets)*100:.1f}%)")


if __name__ == "__main__":
    main()
