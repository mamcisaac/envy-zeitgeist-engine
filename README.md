# Envy Zeitgeist Engine

A **lean, agent‑based pipeline** that ingests raw pop‑culture signals, enriches them with LLM context, and surfaces ranked, ready‑to‑discuss trends for eNVy Media shows (*The Viall Files*, *Ask Nick*, etc.).

---

## ✨ Key Features

| Layer               | Purpose                                                                                                                                                               | Tech                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Collector Agent** | Scrapes Reddit, Twitter/X (free), TikTok, YouTube, entertainment RSS, and direct news sites. De‑duplicates, normalises engagement per hour, writes to `raw_mentions`. | `aiohttp`, `snscrape`, SerpAPI, Supabase (Postgres + PGVector) |
| **Zeitgeist Agent** | Clusters last 24 h mentions, scores momentum + cross‑platform boosts, forecasts peaks, and summarises via GPT‑4o / Claude. Writes to `trending_topics`.               | HDBSCAN, scikit‑learn, ARIMA, OpenAI / Anthropic               |
| **Airflow DAG**     | Runs Collector → Zeitgeist daily (or on‑demand before recording).                                                                                                     | MWAA / Astronomer / docker‑compose Airflow                     |

---

## 🗂️ Repository Structure

```
envy-zeitgeist-engine/
├── pyproject.toml          # Poetry project
├── .env.example            # Copy ➜ .env & fill keys
│
├── envy_toolkit/           # Shared clients & helpers
│   ├── clients.py          # SerpAPI, Reddit, LLM, Supabase
│   ├── schema.py           # RawMention dataclass
│   ├── duplicate.py        # SHA‑256 + PGVector deduper
│   └── twitter_free.py     # Free Twitter/X scraper (snscrape)
│
├── collectors/             # Ingestion modules (async `collect()`)
│   ├── enhanced_celebrity_tracker.py
│   ├── enhanced_network_press_collector.py
│   ├── entertainment_news_collector.py
│   ├── reality_show_controversy_detector.py
│   └── youtube_engagement_collector.py
│
├── agents/
│   ├── collector_agent.py  # Runs all collectors concurrently
│   └── zeitgeist_agent.py  # Trend clustering & briefing
│
├── dags/                   # Airflow DAG(s)
│   └── daily_zeitgeist_dag.py
│
├── supabase/               # SQL migrations
│   └── migrations/001_init.sql
│
└── docker/
    ├── Dockerfile.collector
    └── Dockerfile.zeitgeist
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your‑org/envy‑zeitgeist‑engine.git
cd envy‑zeitgeist‑engine
# Install deps
poetry install --no‑root
# Configure secrets
cp .env.example .env
# edit .env with your keys
```

### Required Environment Variables

| Key                                                               | Description                            |
| ----------------------------------------------------------------- | -------------------------------------- |
| `SERPAPI_API_KEY`                                                 | Google / News search credits           |
| `OPENAI_API_KEY`                                                  | GPT‑4o (embeddings & summaries)        |
| `ANTHROPIC_API_KEY`                                               | Claude‑Sonnet/Opus for long prompts    |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USER_AGENT` | PRAW creds                             |
| `SUPABASE_URL` / `SUPABASE_ANON_KEY`                              | Managed Postgres endpoint              |
| *Optional* `YOUTUBE_API_KEY`                                      | Falls back to unauth scrape if missing |

---

## 🚀 Quick Start (Local)

```bash
# 1️⃣ Run the collector once
poetry run python -m agents.collector_agent

# 2️⃣ Run zeitgeist scoring + briefing
poetry run python -m agents.zeitgeist_agent
```

A Markdown briefing for today's top clusters will now be inserted into `trending_topics`. Query via Supabase SQL or REST.

---

## 🐳 Docker

```bash
# Build images
docker build -f docker/Dockerfile.collector -t envy/collector:latest .
docker build -f docker/Dockerfile.zeitgeist -t envy/zeitgeist:latest .

# One‑off run
docker run --env‑file .env envy/collector:latest
```

Use `docker‑compose` or your orchestrator of choice to schedule daily runs.

---

## 🛫 Airflow Deployment

1. Add `dags/daily_zeitgeist_dag.py` to your Airflow `DAGS_FOLDER`.
2. Ensure the collector & zeitgeist Docker images are available to the workers.
3. Set `AIRFLOW__SECRETS__BACKEND_KWARGS` to pull the same `.env` keys.
4. Trigger manually or let the default cron (`@daily`, 06:00 UTC) kick in.

---

## 🧪 Testing

```bash
ruff .               # lint + formatting check
pytest               # unit tests for collectors & agents
```

---

## 🤝 Contributing

1. Fork & create a feature branch.
2. Add or update **unit tests**.
3. Run `pre‑commit run --all-files`.
4. Open a PR; the CI must pass.

---

## 📄 License

MIT © 2025 eNVy Media LLC. See `LICENSE` for details.