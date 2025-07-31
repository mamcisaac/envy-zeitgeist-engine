-- Enable required extensions
create extension if not exists vector;
create extension if not exists pg_trgm;

-- Raw mentions table
create table if not exists raw_mentions (
    id text primary key,
    source text not null check (source in ('reddit', 'twitter', 'tiktok', 'news', 'youtube')),
    url text not null unique,
    title text not null,
    body text not null,
    timestamp timestamptz not null,
    platform_score numeric not null check (platform_score >= 0),
    embedding vector(1536),
    entities text[] default '{}',
    extras jsonb default '{}',
    created_at timestamptz default now()
);

-- Indexes for performance
create index idx_raw_mentions_timestamp on raw_mentions(timestamp desc);
create index idx_raw_mentions_source on raw_mentions(source);
create index idx_raw_mentions_platform_score on raw_mentions(platform_score desc);
create index idx_raw_mentions_entities on raw_mentions using gin(entities);
create index idx_raw_mentions_embedding on raw_mentions using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Full text search
create index idx_raw_mentions_text_search on raw_mentions using gin(to_tsvector('english', title || ' ' || body));

-- Trending topics table
create table if not exists trending_topics (
    id bigserial primary key,
    created_at timestamptz default now(),
    headline text not null,
    tl_dr text not null,
    score numeric not null check (score >= 0),
    forecast text,
    guests text[] default '{}',
    sample_questions text[] default '{}',
    cluster_ids text[] default '{}',
    extras jsonb default '{}'
);

-- Index for trending topics
create index idx_trending_topics_created_at on trending_topics(created_at desc);
create index idx_trending_topics_score on trending_topics(score desc);

-- View for recent high-engagement mentions
create or replace view hot_mentions as
select 
    id,
    source,
    url,
    title,
    timestamp,
    platform_score,
    entities,
    extras
from raw_mentions
where 
    timestamp > now() - interval '24 hours'
    and platform_score > 50
order by platform_score desc;

-- Function to clean old data (call periodically)
create or replace function cleanup_old_mentions()
returns void as $$
begin
    delete from raw_mentions where timestamp < now() - interval '7 days';
    delete from trending_topics where created_at < now() - interval '30 days';
end;
$$ language plpgsql;