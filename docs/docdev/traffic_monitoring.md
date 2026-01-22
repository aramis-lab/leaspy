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

**View the live dashboard:** [Repository Traffic Dashboard](../traffic_dashboard.md)

## How It Works

### Automated Collection & Publishing (Traffic Branch Architecture)

A GitHub Actions workflow (`.github/workflows/traffic.yaml`) runs every Monday at midnight UTC:

1. Runs from `main` branch (to get latest scripts)
2. Fetches traffic data from GitHub's API
3. Switches to dedicated `traffic` branch
4. Commits CSV files **only to traffic branch** (no binaries)
5. `main` branch stays completely clean (zero data commits)
6. ReadTheDocs builds from `main`, fetches CSVs from `traffic` branch, generates images on-the-fly

### Why This Architecture?

**Problem with old approach:** Committing images (PNGs) weekly would cause "binary bloat" - Git stores entire new files each time, growing the repository significantly over years.

**Solution:** 
- ✅ CSV files (text, small diffs) → `traffic` branch
- ✅ PNG images (large binaries) → generated during build, never committed
- ✅ `main` branch → zero traffic commits, clean history
- ✅ Long-term scalability → repo size stays manageable

### Data Storage

```
two branches:

main branch (code only):
├── traffic/
│   ├── fetch_traffic.py        # Collection script
│   └── visualize_traffic.py    # Visualization script
├── docs/
│   ├── traffic_dashboard.md    # Dashboard page
│   └── _static/traffic/        # Images generated at build time (not in git)
└── .github/workflows/
    └── traffic.yaml            # Workflow (triggers from main, commits to traffic)

traffic branch (data only, orphan):
├── README.md                    # Branch description
└── traffic/data/                # CSV data files
    ├── views.csv
    ├── clones.csv
    ├── referrers.csv
    ├── popular_paths.csv
    └── weekly_summary.csv
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

The dashboard on ReadTheDocs is automatically updated during build. You can also generate visualizations locally:

```bash
# Install dependencies
pip install pandas matplotlib

# Generate from local CSV files (if you have them)
python traffic/visualize_traffic.py --source local

# Generate from GitHub traffic branch (recommended)
python traffic/visualize_traffic.py --source github:aramis-lab/leaspy:traffic

# Filter by date range
python traffic/visualize_traffic.py --source github:aramis-lab/leaspy:traffic 01/01/2026 31/01/2026

# Specify custom output directory
python traffic/visualize_traffic.py --source github:aramis-lab/leaspy:traffic --output-dir /tmp/traffic
```

Generated charts:
- Views and unique visitors over time (line graphs)
- Clones and unique cloners over time (line graphs)
- Weekly traffic summary (bar chart)
- Top referrers (traffic sources)
- Most popular pages

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

## Permissions and Security

### API Access

**Important**: Only users with **admin** or **push** access to the repository can view traffic data via GitHub's API. 

The GitHub Actions workflow uses `GITHUB_TOKEN` which automatically has the necessary permissions when the workflow runs from the default branch.

### Data Protection

- **CSV files** (`traffic/data/*.csv`) are stored in the `traffic` branch and only updated by GitHub Actions
- **PNG images** are generated during ReadTheDocs build from CSVs in `traffic` branch
- **`main` branch** has NO traffic data or images committed (stays clean)
- Manual changes to CSV files should be made in the `traffic` branch (not recommended)

### Git History and Data Commits

The weekly data collection creates one commit per week **in the traffic branch only** with format: `data: update traffic stats YYYY-MM-DD`

**Key benefits of this architecture:**
- ✅ `main` branch history is completely clean (zero traffic commits)
- ✅ No binary bloat (images never committed)
- ✅ Data isolation (traffic branch is independent)
- ✅ Easy to delete entire traffic history if needed (just delete branch)

To view traffic branch:
```bash
# List branches
git branch -a

# Checkout traffic branch
git checkout traffic

# View CSV files
ls traffic/data/

# Return to main
git checkout main
```

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
