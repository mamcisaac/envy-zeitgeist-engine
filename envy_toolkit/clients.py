import os
import asyncio
from typing import List, Dict, Any, Optional
import aiohttp
from dotenv import load_dotenv
from loguru import logger
import openai
import anthropic
from serpapi import search
import praw
from supabase import create_client, Client
import tiktoken

load_dotenv()


class SerpAPIClient:
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY not found in environment")
    
    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": num_results,
            "hl": "en",
            "gl": "us",
            "engine": "google"
        }
        results = search(params)
        return results.get("organic_results", [])
    
    async def search_news(self, query: str) -> List[Dict[str, Any]]:
        params = {
            "q": query,
            "api_key": self.api_key,
            "tbm": "nws",  # News search
            "num": 20,
            "engine": "google"
        }
        results = search(params)
        return results.get("news_results", [])


class RedditClient:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "envy-zeitgeist/0.1")
        )
    
    async def search_subreddit(self, subreddit: str, query: str, limit: int = 100) -> List[Dict[str, Any]]:
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
    def __init__(self):
        self.openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    async def embed_text(self, text: str) -> List[float]:
        response = await self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000]  # Truncate to API limit
        )
        return response.data[0].embedding
    
    async def generate(self, prompt: str, model: str = "gpt-4o", max_tokens: int = 1000) -> str:
        if model.startswith("claude"):
            response = await self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
    
    async def batch(self, prompts: List[str], model: str = "gpt-4o") -> List[str]:
        tasks = [self.generate(p, model) for p in prompts]
        return await asyncio.gather(*tasks)


class SupabaseClient:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY required")
        self.client: Client = create_client(url, key)
    
    async def insert_mention(self, mention: Dict[str, Any]) -> None:
        try:
            self.client.table("raw_mentions").insert(mention).execute()
        except Exception as e:
            logger.error(f"Failed to insert mention: {e}")
    
    async def bulk_insert_mentions(self, mentions: List[Dict[str, Any]]) -> None:
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
        from datetime import datetime, timedelta
        since = datetime.utcnow() - timedelta(hours=hours)
        response = self.client.table("raw_mentions")\
            .select("*")\
            .gte("timestamp", since.isoformat())\
            .execute()
        return response.data
    
    async def insert_trending_topic(self, topic: Dict[str, Any]) -> None:
        self.client.table("trending_topics").insert(topic).execute()


class PerplexityClient:
    """Lightweight Perplexity API wrapper for context enrichment"""
    
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY", os.getenv("OPENAI_API_KEY"))
        self.base_url = "https://api.perplexity.ai" if os.getenv("PERPLEXITY_API_KEY") else None
    
    async def ask(self, question: str) -> str:
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
                    return result["choices"][0]["message"]["content"]
        else:
            # Fallback to GPT-4 with web search prompt
            llm = LLMClient()
            prompt = f"Based on current internet trends and news, {question}"
            return await llm.generate(prompt, model="gpt-4o")