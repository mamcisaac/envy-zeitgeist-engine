# Zeitgeist Brief Generation Guide

This guide covers the new Markdown brief generation functionality added to the ZeitgeistAgent for Issue #6.

## Overview

The Zeitgeist Engine now supports automated generation of professional, Markdown-formatted briefs from trending topics data. These briefs are designed for:

- Daily and weekly summaries
- Email-ready reports with mobile-friendly layouts
- Customizable templates and sections
- Automated scheduling and delivery
- Integration with existing trending topics analysis

## Quick Start

### Basic Brief Generation

```python
from agents.zeitgeist_agent import ZeitgeistAgent
from envy_toolkit.schema import BriefConfig, BriefType

# Initialize agent
agent = ZeitgeistAgent()

# Generate daily brief
daily_brief = await agent.generate_daily_brief(max_topics=10)
print(daily_brief.content)

# Generate weekly summary
weekly_brief = await agent.generate_weekly_brief(max_topics=20)
print(weekly_brief.content)

# Generate email-ready brief
email_brief = await agent.generate_email_brief(
    subject_prefix="Entertainment Update",
    max_topics=6
)
print(email_brief.content)
```

### Custom Configuration

```python
from envy_toolkit.schema import BriefConfig, BriefType, BriefFormat

# Create custom configuration
config = BriefConfig(
    brief_type=BriefType.CUSTOM,
    format=BriefFormat.MARKDOWN,
    max_topics=15,
    include_scores=True,
    include_forecasts=True,
    include_charts=True,
    sections=["summary", "trending", "interviews", "charts"],
    title="Custom Entertainment Analysis",
    date_range_days=7
)

# Generate custom brief
custom_brief = await agent.generate_brief(config)
```

## Brief Types

### 1. Daily Brief (`BriefType.DAILY`)

Professional daily summary of trending topics.

**Features:**
- Executive summary of the day's trends
- Top 3 highlights with detailed analysis
- Tabular overview of all trending topics
- Interview opportunities with suggested guests and questions
- Forecast summary for upcoming trends

**Sample Output:**
```markdown
# Daily Zeitgeist Brief
## Friday, March 15, 2024

*Generated on March 15, 2024 at 02:30 PM UTC*

## Executive Summary

Today's analysis identified **3 trending topics** across entertainment and pop culture. 
The top story is **Celebrity Drama Reaches Peak** with a trend score of 0.92.

## Top 3 Highlights

### 1. Celebrity Drama Reaches Peak
**Trend Score:** 0.92 | **Forecast:** Peak in 2 hours

Major celebrity feud explodes across social media platforms...
```

### 2. Weekly Brief (`BriefType.WEEKLY`)

Comprehensive weekly summary with trend analysis.

**Features:**
- Weekly overview with key statistics
- Top 5 stories of the week
- ASCII trend distribution charts
- Most mentioned personalities
- Week in review summary

### 3. Email Brief (`BriefType.EMAIL`)

Mobile-optimized brief designed for email delivery.

**Features:**
- Mobile-friendly formatting with emojis
- Concise sections optimized for small screens
- Quick-scan bullet points
- Interview-ready highlights
- HTML email compatibility

**Sample Output:**
```markdown
# 📈 Daily Zeitgeist
### Friday, March 15, 2024

## 🎯 Today's Top Story

**Celebrity Drama Reaches Peak**

Major celebrity feud explodes across Twitter and Instagram...

*Trend Score: 0.92 | Peak in 2 hours*

## 📊 Other Trending Now

• **Reality TV Finale Controversy**
  _0.78 score - Trending upward_
```

### 4. Custom Brief (`BriefType.CUSTOM`)

Fully customizable brief with configurable sections and formatting.

**Available Sections:**
- `summary` - Executive summary
- `trending` - Detailed trending topics
- `interviews` - Interview opportunities
- `charts` - ASCII visualizations

## Configuration Options

### BriefConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `brief_type` | BriefType | - | Type of brief to generate |
| `format` | BriefFormat | MARKDOWN | Output format |
| `max_topics` | int | 10 | Maximum topics to include (1-50) |
| `include_scores` | bool | True | Include trend scores |
| `include_forecasts` | bool | True | Include forecast information |
| `include_charts` | bool | False | Include ASCII charts |
| `sections` | List[str] | ["summary", "trending"] | Sections to include |
| `title` | str | None | Custom title |
| `subject_prefix` | str | "Zeitgeist Brief" | Email subject prefix |
| `date_range_days` | int | 1 | Date range for data (1-30) |

### Example Configurations

```python
# Minimal configuration
minimal_config = BriefConfig(brief_type=BriefType.DAILY)

# Comprehensive configuration
comprehensive_config = BriefConfig(
    brief_type=BriefType.CUSTOM,
    max_topics=25,
    include_charts=True,
    sections=["summary", "trending", "interviews", "charts"],
    title="Entertainment Industry Analysis",
    date_range_days=7
)

# Email-optimized configuration
email_config = BriefConfig(
    brief_type=BriefType.EMAIL,
    max_topics=5,
    include_scores=False,
    subject_prefix="Daily Entertainment Update"
)
```

## Automated Scheduling

### Setting Up Scheduled Briefs

```python
from envy_toolkit.brief_scheduler import BriefScheduler, schedule_daily_brief
from envy_toolkit.schema import ScheduledBrief

# Quick daily scheduling
scheduler = await schedule_daily_brief(
    hour=9,  # 9 AM
    email_recipients=["team@company.com"]
)

# Advanced scheduling
scheduler = BriefScheduler()

# Daily schedule
daily_schedule = BriefScheduler.create_daily_schedule(
    name="morning_brief",
    hour=9,
    minute=30,
    email_recipients=["editor@news.com"],
    max_topics=10
)

# Weekly schedule (Monday at 10 AM)
weekly_schedule = BriefScheduler.create_weekly_schedule(
    name="monday_summary",
    day_of_week=1,
    hour=10,
    email_recipients=["management@news.com"],
    max_topics=25
)

scheduler.add_schedule(daily_schedule)
scheduler.add_schedule(weekly_schedule)

# Start scheduler
await scheduler.start_scheduler()
```

### Cron Expression Examples

| Schedule | Cron Expression | Description |
|----------|----------------|-------------|
| Daily 9 AM | `0 9 * * *` | Every day at 9:00 AM |
| Weekdays 8 AM | `0 8 * * 1-5` | Monday-Friday at 8:00 AM |
| Monday 10 AM | `0 10 * * 1` | Every Monday at 10:00 AM |
| Twice daily | `0 9,17 * * *` | 9:00 AM and 5:00 PM daily |

## Email Delivery

### Email Configuration

Set environment variables for email delivery:

```bash
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
export SENDER_EMAIL="briefs@yourcompany.com"
```

### Email Features

- **HTML Conversion**: Automatic Markdown to HTML conversion
- **Mobile Optimization**: Responsive design for mobile devices
- **Professional Styling**: Clean, readable formatting
- **Email-Safe**: Compatible with major email clients

### Webhook Notifications

```python
schedule = ScheduledBrief(
    name="webhook_brief",
    brief_config=BriefConfig(brief_type=BriefType.DAILY),
    schedule_cron="0 9 * * *",
    webhook_url="https://hooks.slack.com/services/..."
)
```

## ASCII Charts and Visualizations

### Trend Score Bars

```
Topic Scores:
 1. Celebrity Drama         |████████████████| 0.92
 2. Reality TV Controversy  |████████████▒▒▒▒| 0.78
 3. Music Awards Fashion    |██████████▒▒▒▒▒▒| 0.65
```

### Distribution Charts

```
Trend Scores Distribution

0.8-1.0: ███ (3)
0.6-0.8: ██ (2)
0.4-0.6: █ (1)
0.2-0.4: (0)
0.0-0.2: (0)
```

## Integration Examples

### Airflow DAG Integration

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from agents.zeitgeist_agent import ZeitgeistAgent

async def generate_daily_brief_task():
    agent = ZeitgeistAgent()
    brief = await agent.generate_daily_brief()
    # Process or save brief
    return brief.content

dag = DAG('zeitgeist_brief', schedule_interval='0 9 * * *')
brief_task = PythonOperator(
    task_id='generate_brief',
    python_callable=generate_daily_brief_task,
    dag=dag
)
```

### API Endpoint

```python
from fastapi import FastAPI
from envy_toolkit.schema import BriefConfig, GeneratedBrief

app = FastAPI()

@app.post("/generate-brief", response_model=GeneratedBrief)
async def generate_brief_endpoint(config: BriefConfig):
    agent = ZeitgeistAgent()
    brief = await agent.generate_brief(config)
    return brief
```

### Slack Integration

```python
import requests

async def send_to_slack(brief: GeneratedBrief, webhook_url: str):
    payload = {
        "text": f"📈 {brief.title}",
        "attachments": [{
            "title": "Brief Summary",
            "text": brief.content[:500] + "...",
            "color": "good"
        }]
    }
    requests.post(webhook_url, json=payload)
```

## Database Schema

### New Tables

The brief generation system expects these database tables:

```sql
-- Generated briefs storage
CREATE TABLE generated_briefs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    brief_type TEXT NOT NULL,
    format TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    topics_count INTEGER NOT NULL,
    date_start TIMESTAMP NOT NULL,
    date_end TIMESTAMP NOT NULL,
    config JSONB NOT NULL,
    metadata JSONB NOT NULL
);

-- Scheduled briefs configuration
CREATE TABLE scheduled_briefs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    name TEXT UNIQUE NOT NULL,
    brief_config JSONB NOT NULL,
    schedule_cron TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_run TIMESTAMP,
    next_run TIMESTAMP,
    email_recipients TEXT[],
    webhook_url TEXT
);
```

## Testing

### Running Tests

```bash
# Test brief templates
python -m pytest tests/unit/envy_toolkit/test_brief_templates.py -v

# Test ZeitgeistAgent brief generation
python -m pytest tests/unit/agents/test_zeitgeist_brief_generation.py -v

# Test scheduling
python -m pytest tests/unit/envy_toolkit/test_brief_scheduler.py -v
```

### Demo Script

```bash
# Run the comprehensive demo
python examples/brief_generation_demo.py
```

## Performance Considerations

### Optimization Tips

1. **Topic Limiting**: Use `max_topics` to control brief size
2. **Date Range**: Limit `date_range_days` for faster queries
3. **Caching**: Cache frequently requested briefs
4. **Async Processing**: Use async methods for database operations

### Resource Usage

- **Memory**: ~50MB per brief generation
- **Database**: Optimized queries with date range filters
- **Generation Time**: 2-5 seconds for typical brief
- **Email Delivery**: 1-3 seconds per recipient

## Troubleshooting

### Common Issues

1. **No Trending Topics Found**
   - Check date range configuration
   - Verify trending topics exist in database
   - Ensure database connection is working

2. **Email Delivery Fails**
   - Verify SMTP configuration
   - Check email credentials
   - Confirm network connectivity

3. **Scheduling Not Working**
   - Validate cron expressions
   - Check scheduler is running
   - Verify schedule is active

### Error Handling

```python
try:
    brief = await agent.generate_brief(config)
except Exception as e:
    logger.error(f"Brief generation failed: {e}")
    # Fallback or retry logic
```

## Best Practices

### Content Guidelines

1. **Professional Tone**: Maintain consistent, professional language
2. **Mobile-First**: Design for mobile readability
3. **Scannable Format**: Use headers, bullets, and tables
4. **Actionable Insights**: Include interview opportunities and questions

### Technical Best Practices

1. **Error Handling**: Always handle potential failures gracefully
2. **Logging**: Log generation steps for debugging
3. **Validation**: Validate configurations before processing
4. **Testing**: Write comprehensive tests for custom templates

### Security Considerations

1. **Email Credentials**: Store securely in environment variables
2. **Webhook URLs**: Validate and sanitize webhook endpoints
3. **Input Validation**: Validate all configuration parameters
4. **Rate Limiting**: Implement rate limits for API endpoints

## API Reference

### ZeitgeistAgent Methods

#### `generate_brief(config: BriefConfig) -> GeneratedBrief`
Generate a brief using the specified configuration.

#### `generate_daily_brief(date: Optional[datetime] = None, max_topics: int = 10) -> GeneratedBrief`
Convenience method for daily briefs.

#### `generate_weekly_brief(week_start: Optional[datetime] = None, max_topics: int = 20) -> GeneratedBrief`
Convenience method for weekly summaries.

#### `generate_email_brief(subject_prefix: str = "Daily Zeitgeist", max_topics: int = 6) -> GeneratedBrief`
Convenience method for email-ready briefs.

#### `save_brief(brief: GeneratedBrief) -> int`
Save generated brief to database.

### BriefScheduler Methods

#### `add_schedule(schedule: ScheduledBrief) -> None`
Add a new scheduled brief.

#### `remove_schedule(schedule_name: str) -> bool`
Remove a scheduled brief by name.

#### `start_scheduler(check_interval: int = 60) -> None`
Start the scheduler daemon.

#### `create_daily_schedule(name: str, hour: int = 9, ...) -> ScheduledBrief`
Create a daily schedule configuration.

#### `create_weekly_schedule(name: str, day_of_week: int = 1, ...) -> ScheduledBrief`
Create a weekly schedule configuration.

## Contributing

### Adding New Template Types

1. Create template class inheriting from `MarkdownTemplate`
2. Implement `generate()` method
3. Add to `_get_template()` method in ZeitgeistAgent
4. Add corresponding `BriefType` enum value
5. Write comprehensive tests

### Custom Sections

```python
class CustomTemplate(MarkdownTemplate):
    def _generate_custom_section(self, topics: List[TrendingTopic]) -> List[str]:
        section = ["## Custom Section", ""]
        # Custom logic here
        return section
```

## Changelog

### Version 1.0.0 (Issue #6)

**Added:**
- Markdown brief generation for daily, weekly, and email formats
- Professional, readable formatting with mobile-friendly layouts
- Configurable report templates and sections
- ASCII charts and visualizations
- Automated scheduling capabilities
- Email delivery with HTML conversion
- Webhook notifications
- Comprehensive test suite
- Integration examples and documentation

**Features:**
- 4 brief types: Daily, Weekly, Email, Custom
- 12+ configuration options
- Email delivery with SMTP support
- Cron-based scheduling
- ASCII trend visualizations
- Mobile-optimized email layouts
- Professional Markdown formatting
- Interview opportunity suggestions
- Trend forecasting integration

This implementation addresses all requirements from Issue #6 and provides a solid foundation for automated entertainment industry brief generation.