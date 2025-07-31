import asyncio
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Set
from loguru import logger
import aiohttp

from envy_toolkit.schema import RawMention
from envy_toolkit.clients import (
    SupabaseClient, LLMClient, SerpAPIClient, 
    RedditClient, PerplexityClient
)
from envy_toolkit.duplicate import DuplicateDetector
from envy_toolkit.twitter_free import collect_twitter


WHITELIST_DOMAINS = {
    "reddit.com", "twitter.com", "x.com", "tiktok.com", "instagram.com",
    "tmz.com", "pagesix.com", "variety.com", "deadline.com", "hollywoodreporter.com",
    "people.com", "usmagazine.com", "eonline.com", "justjared.com", "vulture.com",
    "buzzfeed.com", "popsugar.com", "refinery29.com", "cosmopolitan.com", "elle.com",
    "youtube.com", "dailymail.co.uk", "thesun.co.uk", "mirror.co.uk", "metro.co.uk",
    "nbc.com", "abc.com", "fox.com", "bravotv.com", "wetv.com", "tlc.com",
    "entertainmentweekly.com", "rollingstone.com", "billboard.com", "pitchfork.com"
}

SEED_QUERIES = [
    "celebrity controversy drama scandal today",
    "reality TV fight drama latest",
    "influencer canceled exposed trending",
    "viral TikTok drama celebrity"
]


class CollectorAgent:
    """Main collector agent that orchestrates all data sources"""
    
    def __init__(self):
        self.supabase = SupabaseClient()
        self.llm = LLMClient()
        self.serpapi = SerpAPIClient()
        self.reddit = RedditClient()
        self.perplexity = PerplexityClient()
        self.deduper = DuplicateDetector()
        
    async def run(self) -> None:
        """Main collection pipeline"""
        logger.info("Starting CollectorAgent run")
        
        # Collect from all sources
        raw_mentions = await self._scrape_all_sources()
        logger.info(f"Collected {len(raw_mentions)} raw mentions")
        
        # Validate and clean
        valid_mentions = [m for m in raw_mentions if self._validate_item(m)]
        logger.info(f"Validated {len(valid_mentions)} mentions")
        
        # Deduplicate
        unique_mentions = self.deduper.filter_duplicates(
            [m.model_dump() for m in valid_mentions]
        )
        logger.info(f"After deduplication: {len(unique_mentions)} unique mentions")
        
        # Add embeddings
        enriched_mentions = await self._add_embeddings(unique_mentions)
        
        # Write to database
        await self.supabase.bulk_insert_mentions(enriched_mentions)
        logger.info("CollectorAgent run complete")
    
    def _validate_item(self, item: RawMention) -> bool:
        """Validate that a mention is real and has required data"""
        # Check domain whitelist
        url = item.url or ""
        domain = ""
        if "://" in url:
            parts = url.split("/")
            if len(parts) >= 3:
                domain = parts[2].lower()
        
        if domain not in WHITELIST_DOMAINS:
            logger.debug(f"Rejected URL from unknown domain: {domain}")
            return False
        
        # Must have engagement score
        if not item.platform_score or item.platform_score <= 0:
            logger.debug(f"Rejected item with no engagement score")
            return False
        
        # Must have title and body
        if not item.title or not item.body:
            logger.debug(f"Rejected item with missing title/body")
            return False
        
        # Must be recent (within 48 hours)
        age_hours = (datetime.utcnow() - item.timestamp).total_seconds() / 3600
        if age_hours > 48:
            logger.debug(f"Rejected item older than 48 hours")
            return False
        
        return True
    
    async def _scrape_all_sources(self) -> List[RawMention]:
        """Collect from all configured sources"""
        all_mentions = []
        
        # Get expanded queries
        queries = await self._expand_queries(SEED_QUERIES)
        
        async with aiohttp.ClientSession() as session:
            # Collect from all sources in parallel
            tasks = [
                self._collect_twitter(session),
                self._collect_reddit(queries),
                self._collect_news(queries),
                self._collect_entertainment_sites(session)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Collection task failed: {result}")
                else:
                    all_mentions.extend(result)
        
        return all_mentions
    
    async def _expand_queries(self, seed_queries: List[str]) -> List[str]:
        """Use LLM to expand queries with Gen-Z slang and variations"""
        prompt = """For each query below, provide 2 alternative phrasings that Gen-Z might use on social media.
        Include slang, abbreviations, and trending phrases. Format: one query per line.
        
        Queries:
        {}""".format("\n".join(seed_queries))
        
        try:
            response = await self.llm.generate(prompt, model="gpt-4o", max_tokens=500)
            expanded = response.strip().split("\n")
            all_queries = seed_queries + [q.strip() for q in expanded if q.strip()]
            return list(set(all_queries))  # Remove duplicates
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return seed_queries
    
    async def _collect_twitter(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Collect from Twitter using free scraping"""
        mentions = []
        try:
            async for mention in collect_twitter(session):
                mentions.append(mention)
        except Exception as e:
            logger.error(f"Twitter collection failed: {e}")
        return mentions
    
    async def _collect_reddit(self, queries: List[str]) -> List[RawMention]:
        """Collect from Reddit for entertainment subreddits"""
        subreddits = [
            "entertainment", "popculturechat", "Deuxmoi", "blogsnark",
            "thebachelor", "LoveIslandUSA", "BravoRealHousewives",
            "TeenMomOGandTeenMom2", "KUWTK", "popheads"
        ]
        
        mentions = []
        for sub in subreddits:
            for query in queries[:3]:  # Limit queries per subreddit
                try:
                    posts = await self.reddit.search_subreddit(sub, query, limit=20)
                    for post in posts:
                        # Calculate engagement
                        hours_old = max(
                            (datetime.utcnow().timestamp() - post["created_utc"]) / 3600,
                            1
                        )
                        platform_score = (post["score"] + post["num_comments"] * 2) / hours_old
                        
                        mentions.append(RawMention(
                            id=post["id"],
                            source="reddit",
                            url=post["url"],
                            title=post["title"],
                            body=post["body"] or post["title"],
                            timestamp=datetime.fromtimestamp(post["created_utc"]),
                            platform_score=platform_score,
                            entities=[],  # Will be extracted later
                            extras={"subreddit": sub}
                        ))
                except Exception as e:
                    logger.error(f"Reddit collection failed for r/{sub}: {e}")
        
        return mentions
    
    async def _collect_news(self, queries: List[str]) -> List[RawMention]:
        """Collect from news via SerpAPI"""
        mentions = []
        
        for query in queries:
            try:
                # Search news
                news_results = await self.serpapi.search_news(query + " entertainment celebrity")
                
                for result in news_results[:10]:
                    # Skip if not from whitelist domain
                    link = result.get("link", "")
                    domain = link.split("/")[2] if "://" in link and len(link.split("/")) > 2 else ""
                    if domain not in WHITELIST_DOMAINS:
                        continue
                    
                    # Estimate engagement based on position
                    position_score = 100 / (result.get("position", 1) + 1)
                    
                    mentions.append(RawMention(
                        id=hashlib.sha256(link.encode()).hexdigest(),
                        source="news",
                        url=link,
                        title=result.get("title", ""),
                        body=result.get("snippet", ""),
                        timestamp=datetime.utcnow(),  # News doesn't have exact timestamp
                        platform_score=position_score,
                        entities=[],
                        extras={"source_name": result.get("source", "")}
                    ))
            except Exception as e:
                logger.error(f"News collection failed for query '{query}': {e}")
        
        return mentions
    
    async def _collect_entertainment_sites(self, session: aiohttp.ClientSession) -> List[RawMention]:
        """Direct scraping of entertainment sites (placeholder for actual collectors)"""
        # This is where you would integrate the actual entertainment collectors
        # For now, returning empty list to avoid placeholder data
        return []
    
    async def _add_embeddings(self, mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add OpenAI embeddings to mentions"""
        for mention in mentions:
            try:
                text = f"{mention['title']} {mention['body'][:500]}"
                embedding = await self.llm.embed_text(text)
                mention['embedding'] = embedding
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                mention['embedding'] = None
        
        return mentions


async def main():
    """Run the collector agent"""
    agent = CollectorAgent()
    await agent.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())