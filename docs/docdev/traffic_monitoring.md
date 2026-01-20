---
orphan: true
---
# Traffic Monitoring Guide

This guide explains how Leaspy automatically collects and visualizes repository traffic data.

## Overview

The repository uses GitHub Actions to automatically collect traffic statistics every week. The data includes:

- **Views**: Total page views and unique visitors
- **Clones**: Total clones and unique cloners  
- **Referrers**: Where traffic comes from (search engines, other sites, etc.)
- **Popular Content**: Most visited pages in the repository

## How It Works

### Automated Collection

A GitHub Actions workflow (`.github/workflows/traffic.yaml`) runs every Monday at midnight UTC:

1. Fetches traffic data from GitHub's API
2. Stores it as CSV files in `traffic-data/`
3. Commits the new data automatically
4. Historical data accumulates over time

### Data Storage

All traffic data is stored in CSV format under `traffic/data/`:

```
traffic/
├── data/
│   ├── views.csv           # Daily view counts
│   ├── clones.csv          # Daily clone counts
│   ├── referrers.csv       # Traffic sources
│   ├── popular_paths.csv   # Most viewed pages
│   └── weekly_summary.csv  # Weekly aggregated stats
├── reports/                # Generated visualizations (git-ignored)
├── fetch_traffic.py        # Collection script
└── visualize_traffic.py    # Visualization script
```

## Manual Collection

You can manually trigger data collection:

1. Go to the **Actions** tab on GitHub
2. Select **Repository Traffic Collection** workflow
3. Click **Run workflow** → **Run workflow**

Or run locally (requires admin/push access):

```bash
# Set your GitHub token
export GITHUB_TOKEN=your_token_here
export GITHUB_REPOSITORY=aramis-lab/leaspy

# Fetch data
python traffic/fetch_traffic.py
```

## Generating Visualizations

Create histograms and insights from collected data:

```bash
# Install dependencies (if not already installed)
pip install pandas matplotlib

# Generate all visualizations
python traffic/visualize_traffic.py
```

This creates:

- `traffic/reports/views_timeline.png` - Views over time
- `traffic/reports/clones_timeline.png` - Clones over time
- `traffic/reports/weekly_summary.png` - Weekly comparison
- `traffic/reports/top_referrers.png` - Top traffic sources
- `traffic/reports/popular_paths.png` - Most visited pages

Plus a text summary with key insights.

## Understanding the Data

### Views vs Clones

- **Views**: Someone visited the repository on GitHub (via browser)
- **Clones**: Someone ran `git clone` to download the repository

### Unique vs Total

- **Total**: All events (includes repeat visits from same user)
- **Unique**: Distinct users (GitHub tracks by IP/session)

### Data Retention

GitHub only provides the last 14 days of detailed traffic data via the API. This is why **weekly automated collection is crucial** - it preserves historical data beyond GitHub's 14-day window.

## Common Use Cases

### Analyzing Growth Trends

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load historical data
df = pd.read_csv('traffic/data/weekly_summary.csv')
df['fetch_timestamp'] = pd.to_datetime(df['fetch_timestamp'])

# Plot growth
plt.plot(df['fetch_timestamp'], df['unique_visitors'])
plt.title('Visitor Growth Over Time')
plt.show()
```

### Finding Peak Traffic Days

```python
import pandas as pd

df = pd.read_csv('traffic/data/views.csv')
df['date'] = pd.to_datetime(df['date'])

# Find highest traffic day
peak = df.loc[df['count'].idxmax()]
print(f"Peak traffic: {peak['count']} views on {peak['date']}")
```

### Identifying Traffic Sources

```python
import pandas as pd

df = pd.read_csv('traffic/data/referrers.csv')

# Aggregate all referrers
referrers = df.groupby('referrer')['count'].sum().sort_values(ascending=False)
print("\nTop traffic sources:")
print(referrers.head(10))
```

## Permissions

**Important**: Only users with **admin** or **push** access to the repository can view traffic data via GitHub's API. 

The GitHub Actions workflow uses `GITHUB_TOKEN` which automatically has the necessary permissions when the workflow runs from the default branch.

## Troubleshooting

### Workflow Not Running

**Problem**: Scheduled workflow doesn't trigger weekly

**Solution**: Scheduled workflows only run from the **default branch** (`master` or `main`). Merge your branch into the default branch.

### No Data Collected

**Problem**: Workflow runs but no data is saved

**Solution**: The repository may not have received any traffic in the last 14 days, or the `GITHUB_TOKEN` doesn't have sufficient permissions.

### Visualization Errors

**Problem**: `visualize_traffic.py` fails

**Solution**: Ensure dependencies are installed:
```bash
pip install pandas matplotlib
```

## Adding Custom Analysis

You can extend the monitoring system:

1. **Custom metrics**: Edit `traffic/fetch_traffic.py` to fetch additional data
2. **Custom visualizations**: Add new plot functions to `traffic/visualize_traffic.py`
3. **Alerts**: Add notification logic (e.g., send email if traffic drops)

Example: Add a plot comparing weekday vs weekend traffic:

```python
def plot_weekday_analysis(data: dict, output_dir: Path):
    """Compare weekday vs weekend traffic."""
    df = data['views'].copy()
    df['weekday'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['weekday'] >= 5
    
    weekend = df[df['is_weekend']]['count'].mean()
    weekday = df[~df['is_weekend']]['count'].mean()
    
    plt.bar(['Weekday', 'Weekend'], [weekday, weekend])
    plt.title('Average Traffic: Weekday vs Weekend')
    plt.ylabel('Average Views')
    plt.savefig(output_dir / 'weekday_analysis.png')
```

## Best Practices

1. **Run weekly**: The current schedule (Monday midnight) captures full weeks
2. **Don't edit CSV files manually**: Let automation handle it
3. **Archive old data**: If CSV files get very large, consider archiving older data
4. **Review regularly**: Check visualizations monthly to spot trends
5. **Share insights**: Use the data to inform documentation improvements

## Related Resources

- [GitHub Traffic API Documentation](https://docs.github.com/en/rest/metrics/traffic)
- [GitHub Actions Scheduled Events](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule)
