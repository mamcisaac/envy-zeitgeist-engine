"""
LLM-based story clustering for better semantic grouping
"""

import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, List

from loguru import logger

from .clients import LLMClient


class LLMStoryClustering:
    """Use LLM to intelligently group posts into coherent stories."""

    def __init__(self):
        self.llm = LLMClient()

    async def cluster_stories(self, posts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group posts into stories using LLM intelligence.
        
        Returns:
            Dict mapping story_id -> list of posts in that story
        """
        if not posts:
            return {}

        # Prepare posts for LLM analysis
        posts_summary = []
        for i, post in enumerate(posts):
            posts_summary.append({
                "index": i,
                "platform": post.get("platform", "unknown"),
                "title": post.get("title", "")[:100],
                "body": post.get("body", "")[:200],
                "url": post.get("url", "")
            })

        # Ask LLM to group posts into stories
        clustering_prompt = f"""Analyze these social media posts and group them into distinct stories/narratives.

Posts:
{json.dumps(posts_summary, indent=2)}

Group posts that are about the same story, event, or topic. Consider:
- Same people/characters involved
- Same event/incident
- Related discussions about the same topic

Return a JSON object where:
- Keys are story identifiers (brief description like "story_1")
- Values are arrays of post indices that belong to that story

Example output:
{{
  "story_1": [0, 2, 4, 7],
  "story_2": [1, 3],
  "story_3": [5, 6]
}}

Only return the JSON object, no other text."""

        try:
            response = await self.llm.complete(clustering_prompt, model="claude-3-5-sonnet-20241022")

            # Parse LLM response
            clusters = json.loads(response.strip())

            # Convert indices back to posts
            story_groups = {}
            for story_id, indices in clusters.items():
                story_posts = []
                for idx in indices:
                    if 0 <= idx < len(posts):
                        story_posts.append(posts[idx])
                if story_posts:
                    story_groups[story_id] = story_posts

            logger.info(f"LLM clustered {len(posts)} posts into {len(story_groups)} stories")
            return story_groups

        except Exception as e:
            logger.error(f"LLM clustering failed: {e}")
            # Fallback: group by URL similarity
            return self._fallback_url_clustering(posts)

    def _fallback_url_clustering(self, posts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Simple URL-based clustering as fallback."""
        url_groups = defaultdict(list)

        for post in posts:
            # Extract base URL or create one from title
            url = post.get("url", "")
            if not url:
                # Create pseudo-URL from title
                title_hash = hashlib.md5(post.get("title", "").encode()).hexdigest()[:8]
                url = f"no-url-{title_hash}"

            # Group by base URL
            base_url = url.split("?")[0].split("#")[0]
            url_groups[base_url].append(post)

        return dict(url_groups)

    async def get_story_summaries(self, story_groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """
        Generate concise summaries for each story cluster.
        
        Returns:
            Dict mapping story_id -> one-line summary
        """
        summaries = {}

        for story_id, posts in story_groups.items():
            # Get key details from posts
            platforms = list(set(p.get("platform", "unknown") for p in posts))
            titles = [p.get("title", "")[:100] for p in posts[:3]]  # First 3 titles

            summary_prompt = f"""Based on these social media posts about the same story, write a single concise headline (max 100 chars).

Story ID: {story_id}
Platforms: {platforms}
Sample titles:
{chr(10).join(f'- {t}' for t in titles)}

Write a specific, newsworthy headline that captures the story. Include names and key details.
Examples of good headlines:
- "JaNa and Kenny confirm Love Island breakup after Casa Amor drama"
- "Marcus self-eliminates from Bachelorette after tense hometown date"

Return only the headline text, nothing else."""

            try:
                summary = await self.llm.complete(summary_prompt, model="claude-3-5-sonnet-20241022")
                summaries[story_id] = summary.strip()
            except Exception as e:
                logger.error(f"Failed to generate summary for {story_id}: {e}")
                # Fallback: use first title
                summaries[story_id] = posts[0].get("title", story_id)[:100]

        return summaries


# Global instance
llm_story_clustering = LLMStoryClustering()
