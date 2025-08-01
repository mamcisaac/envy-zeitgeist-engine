import asyncio
import hashlib
from datetime import datetime
from typing import Any, Dict, List

import aiohttp
from loguru import logger

from envy_toolkit.clients import (
    LLMClient,
    PerplexityClient,
    RedditClient,
    SerpAPIClient,
    SupabaseClient,
)
from envy_toolkit.duplicate import DuplicateDetector
from envy_toolkit.error_handler import handle_errors
from envy_toolkit.exceptions import (
    DataCollectionError,
    ProcessingError,
    ValidationError,
)
from envy_toolkit.logging_config import LogContext
from envy_toolkit.metrics import collect_metrics, get_metrics_collector
from envy_toolkit.schema import RawMention
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
    """Main collector agent that orchestrates zeitgeist data collection.

    Collects mentions from multiple sources (Twitter, Reddit, news, entertainment sites),
    validates and deduplicates them, adds embeddings, and stores them to the database.
    Implements robust error handling, metrics collection, and graceful degradation.

    Attributes:
        supabase: Database client for storing collected data
        llm: LLM client for embeddings and query expansion
        serpapi: SerpAPI client for news search
        reddit: Reddit client for subreddit searches
        perplexity: Perplexity client for context enrichment
        deduper: Duplicate detection utility

    Example:
        >>> agent = CollectorAgent()
        >>> await agent.run()
    """

    def __init__(self) -> None:
        self.supabase = SupabaseClient()
        self.llm = LLMClient()
        self.serpapi = SerpAPIClient()
        self.reddit = RedditClient()
        self.perplexity = PerplexityClient()
        self.deduper = DuplicateDetector()

    @collect_metrics(operation_name="collector_agent_run")
    @handle_errors(operation_name="collector_agent_run")
    async def run(self) -> None:
        """Main collection pipeline with enhanced error handling and metrics."""
        with LogContext(operation="collector_agent_run"):
            logger.info("Starting CollectorAgent run")

        metrics = get_metrics_collector()
        metrics.increment_counter("collector_agent_runs")

        # Collect from all sources with error handling
        try:
            raw_mentions = await self._scrape_all_sources()
            with LogContext(raw_mentions_count=len(raw_mentions)):
                logger.info(f"Collected {len(raw_mentions)} raw mentions")
            metrics.set_gauge("raw_mentions_collected", len(raw_mentions))
        except Exception as e:
            raise DataCollectionError(
                "Failed to collect raw mentions from sources",
                cause=e,
                context={"sources": "all"}
            )

        # Validate and clean with metrics
        try:
            valid_mentions = [m for m in raw_mentions if self._validate_item(m)]
            validation_rate = len(valid_mentions) / len(raw_mentions) if raw_mentions else 0
            with LogContext(valid_mentions_count=len(valid_mentions), validation_rate=validation_rate):
                logger.info(f"Validated {len(valid_mentions)} mentions (rate: {validation_rate:.2%})")
            metrics.set_gauge("valid_mentions_count", len(valid_mentions))
            metrics.observe_histogram("validation_rate", validation_rate)
        except Exception as e:
            raise ValidationError(
                "Failed to validate collected mentions",
                cause=e,
                context={"raw_count": len(raw_mentions)}
            )

        # Deduplicate with error handling
        try:
            unique_mentions = self.deduper.filter_duplicates(
                [m.model_dump() for m in valid_mentions]
            )
            duplicates_removed = len(valid_mentions) - len(unique_mentions)
            with LogContext(unique_mentions_count=len(unique_mentions), duplicates_removed=duplicates_removed):
                logger.info(f"After deduplication: {len(unique_mentions)} unique mentions ({duplicates_removed} duplicates removed)")
            metrics.set_gauge("unique_mentions_count", len(unique_mentions))
            metrics.increment_counter("duplicates_removed", duplicates_removed)
        except Exception as e:
            raise ProcessingError(
                "Failed to deduplicate mentions",
                operation="deduplication",
                cause=e,
                context={"valid_count": len(valid_mentions)}
            )

        # Add embeddings with error handling
        try:
            enriched_mentions = await self._add_embeddings(unique_mentions)
            with LogContext(enriched_mentions_count=len(enriched_mentions)):
                logger.info(f"Added embeddings to {len(enriched_mentions)} mentions")
            metrics.set_gauge("enriched_mentions_count", len(enriched_mentions))
        except Exception:
            # Use fallback without embeddings for graceful degradation
            logger.warning("Failed to add embeddings, proceeding without embeddings", exc_info=True)
            enriched_mentions = unique_mentions
            metrics.increment_counter("embedding_failures")

        # Write to database with error handling
        try:
            await self.supabase.bulk_insert_mentions(enriched_mentions)
            with LogContext(stored_mentions_count=len(enriched_mentions)):
                logger.info(f"Successfully stored {len(enriched_mentions)} mentions to database")
            metrics.increment_counter("mentions_stored", len(enriched_mentions))
        except Exception as e:
            raise DataCollectionError(
                "Failed to store mentions to database",
                cause=e,
                context={"mention_count": len(enriched_mentions)}
            )

        with LogContext(operation="collector_agent_run", final_count=len(enriched_mentions)):
            logger.info("CollectorAgent run complete")
        metrics.increment_counter("collector_agent_runs_completed")

    def _validate_item(self, item: RawMention) -> bool:
        """Validate that a mention meets quality and recency requirements.

        Checks domain whitelist, engagement scores, required fields,
        and recency (must be within 48 hours) to ensure data quality.

        Args:
            item: Raw mention to validate

        Returns:
            True if the mention passes validation, False otherwise

        Example:
            >>> mention = RawMention(title="News", body="Content", ...)
            >>> is_valid = agent._validate_item(mention)
        """
        # Check domain whitelist
        url = item.url or ""
        domain = ""
        if "://" in url:
            parts = url.split("/")
            if len(parts) >= 3:
                domain = parts[2].lower()
                # Remove www. prefix if present
                if domain.startswith("www."):
                    domain = domain[4:]

        if domain not in WHITELIST_DOMAINS:
            logger.debug(f"Rejected URL from unknown domain: {domain}")
            return False

        # Must have engagement score
        if not item.platform_score or item.platform_score <= 0:
            logger.debug("Rejected item with no engagement score")
            return False

        # Must have title and body
        if not item.title or not item.body:
            logger.debug("Rejected item with missing title/body")
            return False

        # Must be recent (within 48 hours)
        age_hours = (datetime.utcnow() - item.timestamp).total_seconds() / 3600
        if age_hours > 48:
            logger.debug("Rejected item older than 48 hours")
            return False

        return True

    async def _scrape_all_sources(self) -> List[RawMention]:
        """Collect mentions from all configured data sources in parallel.

        Coordinates collection from Twitter, Reddit, news APIs, and entertainment
        sites concurrently. Uses expanded queries and handles individual source
        failures gracefully without stopping the entire collection.

        Returns:
            List of raw mentions collected from all sources

        Note:
            Failed source collections are logged but don't stop other sources.
        """
        all_mentions: List[RawMention] = []

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
                elif isinstance(result, list):
                    all_mentions.extend(result)

        return all_mentions

    async def _expand_queries(self, seed_queries: List[str]) -> List[str]:
        """Expand search queries using LLM to include current slang and variations.

        Uses GPT-4 to generate alternative phrasings of seed queries with
        Gen-Z slang, abbreviations, and trending phrases to improve coverage
        of social media content.

        Args:
            seed_queries: Base search queries to expand

        Returns:
            Combined list of original and expanded queries with duplicates removed

        Note:
            Falls back to original queries if LLM expansion fails.
        """
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
        """Collect Twitter mentions using free scraping methods.

        Uses the twitter_free module to scrape Twitter content without
        requiring official API access. Handles rate limiting and errors gracefully.

        Args:
            session: Async HTTP session for making requests

        Returns:
            List of raw mentions collected from Twitter

        Note:
            Failures are logged but don't raise exceptions to avoid breaking collection.
        """
        mentions = []
        try:
            async for mention in collect_twitter(session):
                mentions.append(mention)
        except Exception as e:
            logger.error(f"Twitter collection failed: {e}")
        return mentions

    async def _collect_reddit(self, queries: List[str]) -> List[RawMention]:
        """Collect mentions from entertainment-focused subreddits.

        Searches curated list of entertainment subreddits using provided queries.
        Calculates engagement scores based on post score, comments, and age.
        Limits queries per subreddit to avoid rate limiting.

        Args:
            queries: List of search query strings

        Returns:
            List of raw mentions from Reddit posts

        Note:
            Uses time-decay scoring where newer posts get higher engagement weights.
        """
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
        """Collect news articles using SerpAPI Google News search.

        Searches Google News for each query with entertainment focus.
        Filters results to whitelisted domains and calculates engagement
        scores based on search result position.

        Args:
            queries: List of search query strings

        Returns:
            List of raw mentions from news articles

        Note:
            Only includes results from domains in WHITELIST_DOMAINS.
        """
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
        """Collect from specialized entertainment site collectors.

        Runs all registered collectors from the collectors module in parallel.
        Each collector targets specific entertainment websites and implements
        domain-specific scraping logic.

        Args:
            session: Async HTTP session for making requests

        Returns:
            Combined list of mentions from all entertainment site collectors

        Note:
            Individual collector failures don't stop other collectors.
        """
        from collectors import registry

        all_mentions: List[RawMention] = []

        # Run all collectors in parallel
        tasks = [collector(session) for collector in registry]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Collector {i} failed: {result}")
            elif isinstance(result, list):
                all_mentions.extend(result)
                logger.info(f"Collector {i} returned {len(result)} mentions")

        return all_mentions

    async def _add_embeddings(self, mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add vector embeddings to mentions for similarity analysis.

        Generates OpenAI embeddings for each mention's title and body text
        (truncated to 500 chars). Failed embeddings are set to None to allow
        graceful degradation.

        Args:
            mentions: List of mention dictionaries to enrich

        Returns:
            List of mentions with 'embedding' field added

        Note:
            Individual embedding failures are logged but don't stop processing.
        """
        for mention in mentions:
            try:
                text = f"{mention['title']} {mention['body'][:500]}"
                embedding = await self.llm.embed_text(text)
                mention['embedding'] = embedding
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                mention['embedding'] = None

        return mentions


async def main() -> None:
    """Run the collector agent pipeline.

    Entry point for running the complete data collection pipeline.
    Creates a CollectorAgent instance and executes the full collection,
    validation, deduplication, and storage workflow.

    Example:
        >>> await main()
    """
    agent = CollectorAgent()
    await agent.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
