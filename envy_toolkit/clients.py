import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List

import aiohttp
import anthropic
import openai
import praw
import tiktoken
from dotenv import load_dotenv
from loguru import logger
from serpapi import search

from supabase import Client, create_client

load_dotenv()


class SerpAPIClient:
    """Client for interacting with SerpAPI to search Google results.

    Provides methods to search Google web results and Google News for
    zeitgeist data collection. Handles API authentication and result formatting.

    Attributes:
        api_key: SerpAPI key for authentication

    Raises:
        ValueError: If SERPAPI_API_KEY environment variable is not set
    """

    def __init__(self) -> None:
        """Initialize SerpAPI client with API key from environment.

        Raises:
            ValueError: If SERPAPI_API_KEY environment variable is not found
        """
        self.api_key = os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY not found in environment")

    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Search Google web results using SerpAPI.

        Performs a general Google search and returns organic results.
        Results are filtered to ensure they are in list format.

        Args:
            query: Search query string
            num_results: Maximum number of results to return (default: 10)

        Returns:
            List of dictionaries containing search results with keys like
            'title', 'link', 'snippet', 'position', etc.

        Example:
            >>> client = SerpAPIClient()
            >>> results = await client.search("celebrity news")
            >>> print(f"Found {len(results)} results")
        """
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": num_results,
            "hl": "en",
            "gl": "us",
            "engine": "google"
        }
        results = search(params)
        organic_results = results.get("organic_results", [])
        return organic_results if isinstance(organic_results, list) else []

    async def search_news(self, query: str) -> List[Dict[str, Any]]:
        """Search Google News results using SerpAPI.

        Performs a Google News search to find recent news articles
        related to the query. Limited to 20 results for performance.

        Args:
            query: News search query string

        Returns:
            List of dictionaries containing news results with keys like
            'title', 'link', 'snippet', 'date', 'source', etc.

        Example:
            >>> client = SerpAPIClient()
            >>> news = await client.search_news("celebrity controversy")
            >>> for article in news[:3]:
            ...     print(f"{article['title']} - {article['source']}")
        """
        params = {
            "q": query,
            "api_key": self.api_key,
            "tbm": "nws",  # News search
            "num": 20,
            "engine": "google"
        }
        results = search(params)
        news_results = results.get("news_results", [])
        return news_results if isinstance(news_results, list) else []


class RedditClient:
    """Client for interacting with Reddit API using PRAW.

    Provides methods to search subreddits for posts related to entertainment
    and celebrity news. Handles Reddit authentication and result formatting.

    Attributes:
        reddit: PRAW Reddit instance for API interactions

    Note:
        Requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET environment variables.
        REDDIT_USER_AGENT is optional (defaults to 'envy-zeitgeist/0.1').
    """

    def __init__(self) -> None:
        """Initialize Reddit client with PRAW configuration.

        Uses environment variables for authentication:
        - REDDIT_CLIENT_ID: Reddit app client ID
        - REDDIT_CLIENT_SECRET: Reddit app client secret
        - REDDIT_USER_AGENT: User agent string (optional)
        """
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "envy-zeitgeist/0.1")
        )

    async def search_subreddit(self, subreddit: str, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search for posts in a specific subreddit.

        Searches the specified subreddit for posts matching the query,
        sorted by 'hot' and filtered to posts from the last day.

        Args:
            subreddit: Name of the subreddit to search (without 'r/')
            query: Search query string
            limit: Maximum number of posts to return (default: 100)

        Returns:
            List of dictionaries containing post data with keys:
            - id: Reddit post ID
            - title: Post title
            - body: Post text content (selftext)
            - url: Full Reddit URL to the post
            - score: Post score (upvotes - downvotes)
            - num_comments: Number of comments
            - created_utc: UTC timestamp of post creation

        Example:
            >>> client = RedditClient()
            >>> posts = await client.search_subreddit("entertainment", "celebrity news")
            >>> for post in posts[:3]:
            ...     print(f"{post['title']} (Score: {post['score']})")
        """
        sub = self.reddit.subreddit(subreddit)
        posts = []
        for post in sub.search(query, limit=limit, sort="hot", time_filter="day"):
            posts.append({
                "id": post.id,
                "title": post.title,
                "body": post.selftext,
                "url": f"https://reddit.com{post.permalink}",
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": post.created_utc
            })
        return posts


class LLMClient:
    """Multi-provider LLM client supporting OpenAI and Anthropic models.

    Provides unified interface for text generation, embeddings, and batch processing
    across different LLM providers. Automatically routes requests based on model name.

    Attributes:
        openai_client: AsyncOpenAI client for OpenAI models
        anthropic_client: AsyncAnthropic client for Claude models
        tokenizer: Tiktoken encoder for token counting

    Note:
        Requires OPENAI_API_KEY and/or ANTHROPIC_API_KEY environment variables.
    """

    def __init__(self) -> None:
        """Initialize LLM client with OpenAI and Anthropic APIs.

        Sets up async clients for both providers and initializes
        the tiktoken tokenizer for OpenAI-compatible token counting.
        """
        self.openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def embed_text(self, text: str) -> List[float]:
        """Generate text embeddings using OpenAI's embedding model.

        Creates vector embeddings for the input text using the
        text-embedding-3-small model. Text is truncated to 8000 characters
        to stay within API limits.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the text embedding vector

        Raises:
            openai.OpenAIError: If the API request fails

        Example:
            >>> client = LLMClient()
            >>> embedding = await client.embed_text("celebrity news article text")
            >>> print(f"Embedding dimension: {len(embedding)}")
        """
        response = await self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000]  # Truncate to API limit
        )
        return response.data[0].embedding

    async def generate(self, prompt: str, model: str = "gpt-4o", max_tokens: int = 1000) -> str:
        """Generate text using OpenAI or Anthropic models.

        Routes the request to the appropriate provider based on model name.
        Claude models (starting with 'claude') use Anthropic API,
        all other models use OpenAI API.

        Args:
            prompt: Input prompt for text generation
            model: Model name (e.g., 'gpt-4o', 'claude-3-sonnet-20240229')
            max_tokens: Maximum tokens to generate (default: 1000)

        Returns:
            Generated text response as string

        Raises:
            openai.OpenAIError: If OpenAI API request fails
            anthropic.AnthropicError: If Anthropic API request fails

        Example:
            >>> client = LLMClient()
            >>> response = await client.generate("Summarize this celebrity news")
            >>> print(response)
        """
        if model.startswith("claude"):
            response = await self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            # Extract text from the response content
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
            return ""  # Fallback if no text block found
        else:
            openai_response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            content = openai_response.choices[0].message.content
            return content if content is not None else ""

    async def batch(self, prompts: List[str], model: str = "gpt-4o") -> List[str]:
        """Process multiple prompts concurrently.

        Sends multiple prompts to the LLM concurrently for improved throughput.
        All prompts use the same model and default parameters.

        Args:
            prompts: List of prompt strings to process
            model: Model name to use for all prompts (default: 'gpt-4o')

        Returns:
            List of generated text responses in the same order as input prompts

        Raises:
            openai.OpenAIError: If any OpenAI API requests fail
            anthropic.AnthropicError: If any Anthropic API requests fail

        Example:
            >>> client = LLMClient()
            >>> prompts = ["Summarize article 1", "Summarize article 2"]
            >>> responses = await client.batch(prompts)
            >>> for i, response in enumerate(responses):
            ...     print(f"Summary {i+1}: {response}")
        """
        tasks = [self.generate(p, model) for p in prompts]
        return await asyncio.gather(*tasks)


class SupabaseClient:
    """Client for interacting with Supabase database.

    Provides methods for storing and retrieving zeitgeist data including
    raw mentions, trending topics, and briefs. Handles connection management
    and error handling for database operations.

    Attributes:
        client: Supabase client instance for database operations

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_ANON_KEY environment variables are not set
    """

    def __init__(self) -> None:
        """Initialize Supabase client with URL and API key from environment.

        Raises:
            ValueError: If required environment variables are missing
        """
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY required")
        self.client: Client = create_client(url, key)

    async def insert_mention(self, mention: Dict[str, Any]) -> None:
        """Insert a single raw mention into the database.

        Stores a collected mention in the raw_mentions table.
        Errors are logged but not raised to avoid breaking the collection pipeline.

        Args:
            mention: Dictionary containing mention data matching the raw_mentions schema

        Example:
            >>> client = SupabaseClient()
            >>> mention = {
            ...     "id": "tweet_123",
            ...     "source": "twitter",
            ...     "title": "Celebrity news",
            ...     "body": "Breaking news about...",
            ...     "url": "https://twitter.com/status/123",
            ...     "timestamp": "2023-01-01T00:00:00Z",
            ...     "platform_score": 100.0
            ... }
            >>> await client.insert_mention(mention)
        """
        try:
            self.client.table("raw_mentions").insert(mention).execute()
        except Exception as e:
            logger.error(f"Failed to insert mention: {e}")

    async def bulk_insert_mentions(self, mentions: List[Dict[str, Any]]) -> None:
        """Insert multiple raw mentions in batches.

        Efficiently inserts large numbers of mentions by batching them
        into chunks of 100 to avoid API limits. Skips empty lists.

        Args:
            mentions: List of mention dictionaries to insert

        Example:
            >>> client = SupabaseClient()
            >>> mentions = [mention1, mention2, mention3]  # List of mention dicts
            >>> await client.bulk_insert_mentions(mentions)
        """
        if not mentions:
            return
        try:
            # Batch insert in chunks of 100
            for i in range(0, len(mentions), 100):
                batch = mentions[i:i+100]
                self.client.table("raw_mentions").insert(batch).execute()
            logger.info(f"Inserted {len(mentions)} mentions")
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")

    async def get_recent_mentions(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Retrieve recent mentions from the database.

        Fetches all mentions from the specified time window,
        ordered by timestamp descending.

        Args:
            hours: Number of hours back to search (default: 24)

        Returns:
            List of mention dictionaries from the raw_mentions table

        Example:
            >>> client = SupabaseClient()
            >>> recent = await client.get_recent_mentions(hours=12)
            >>> print(f"Found {len(recent)} mentions from last 12 hours")
        """
        from datetime import datetime, timedelta
        since = datetime.utcnow() - timedelta(hours=hours)
        response = self.client.table("raw_mentions")\
            .select("*")\
            .gte("timestamp", since.isoformat())\
            .execute()
        return response.data if isinstance(response.data, list) else []

    async def insert_trending_topic(self, topic: Dict[str, Any]) -> None:
        """Insert a trending topic into the database.

        Stores an analyzed trending topic in the trending_topics table.

        Args:
            topic: Dictionary containing trending topic data matching the schema

        Example:
            >>> client = SupabaseClient()
            >>> topic = {
            ...     "headline": "Celebrity Scandal Trending",
            ...     "tl_dr": "Multiple sources reporting...",
            ...     "score": 85.5,
            ...     "forecast": "Peak in 3-6 hours",
            ...     "guests": ["Entertainment Reporter"],
            ...     "sample_questions": ["What's your take?"]
            ... }
            >>> await client.insert_trending_topic(topic)
        """
        self.client.table("trending_topics").insert(topic).execute()

    async def get_trending_topics_by_date_range(self, start_date: datetime, end_date: datetime, limit: int = 50) -> List[Dict[str, Any]]:
        """Get trending topics within a specified date range.

        Retrieves trending topics from the database that were created within
        the specified date range, ordered by score descending.

        Args:
            start_date: Start of the date range (inclusive)
            end_date: End of the date range (inclusive)
            limit: Maximum number of topics to return (default: 50)

        Returns:
            List of trending topic dictionaries from the database

        Example:
            >>> client = SupabaseClient()
            >>> start = datetime(2024, 1, 1)
            >>> end = datetime(2024, 1, 7)
            >>> topics = await client.get_trending_topics_by_date_range(start, end, 10)
            >>> print(f"Found {len(topics)} trending topics")
        """
        response = self.client.table("trending_topics")\
            .select("*")\
            .gte("created_at", start_date.isoformat())\
            .lte("created_at", end_date.isoformat())\
            .order("score", desc=True)\
            .limit(limit)\
            .execute()
        return response.data if isinstance(response.data, list) else []


class PerplexityClient:
    """Client for Perplexity AI API with fallback to OpenAI.

    Provides access to Perplexity's online-capable models for enriching
    zeitgeist data with current context. Falls back to OpenAI GPT-4 with
    web search prompts if Perplexity API key is not available.

    Attributes:
        api_key: API key for Perplexity or OpenAI
        base_url: Perplexity API base URL (None if using fallback)
    """

    def __init__(self) -> None:
        """Initialize Perplexity client with API configuration.

        Uses PERPLEXITY_API_KEY if available, otherwise falls back to
        OPENAI_API_KEY for GPT-4 with web search prompts.
        """
        self.api_key = os.getenv("PERPLEXITY_API_KEY", os.getenv("OPENAI_API_KEY"))
        self.base_url = "https://api.perplexity.ai" if os.getenv("PERPLEXITY_API_KEY") else None

    async def ask(self, question: str) -> str:
        """Ask a question using Perplexity AI or GPT-4 fallback.

        Routes the question to Perplexity's online-capable model if API key
        is available, otherwise uses OpenAI GPT-4 with a web search prompt.

        Args:
            question: Question to ask about current events or trends

        Returns:
            Answer string from the AI model

        Raises:
            aiohttp.ClientError: If Perplexity API request fails
            openai.OpenAIError: If fallback OpenAI request fails

        Example:
            >>> client = PerplexityClient()
            >>> answer = await client.ask("What's trending in celebrity news today?")
            >>> print(answer)
        """
        if self.base_url:
            # Use actual Perplexity API
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                data = {
                    "model": "pplx-70b-online",
                    "messages": [{"role": "user", "content": question}]
                }
                async with session.post(f"{self.base_url}/chat/completions",
                                       headers=headers, json=data) as resp:
                    result = await resp.json()
                    content = result["choices"][0]["message"]["content"]
                    return content if isinstance(content, str) else ""
        else:
            # Fallback to GPT-4 with web search prompt
            llm = LLMClient()
            prompt = f"Based on current internet trends and news, {question}"
            return await llm.generate(prompt, model="gpt-4o")
