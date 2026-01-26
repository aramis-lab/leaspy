#!/usr/bin/env python3
"""
Fetch GitHub repository traffic data and store it as CSV.

This script collects:
- Views (total and unique)
- Clones (total and unique)
- Top referrers
- Popular content (paths)

Data is saved to traffic/data/ directory locally, but committed to traffic branch by GitHub Actions.
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd


def get_traffic_data(repo: str, token: str, endpoint: str) -> dict:
    """Fetch traffic data from GitHub API.
    
    Args:
        repo: Repository in 'owner/name' format
        token: GitHub personal access token
        endpoint: API endpoint (e.g., 'views', 'clones', 'popular/referrers')
        
    Returns:
        Dictionary containing traffic data, or empty dict on error
    """
    url = f"https://api.github.com/repos/{repo}/traffic/{endpoint}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 403:
        print(f"Warning: No access to {endpoint} data. Requires admin or push access to repository.")
        return {}
    else:
        print(f"Warning: Failed to fetch {endpoint} (HTTP {response.status_code})")
        return {}


def save_timeseries_data(data: dict, metric_name: str, output_dir: Path):
    """Save time-series traffic data (views/clones) to CSV.
    
    Args:
        data: Dictionary containing traffic metrics from GitHub API
        metric_name: Name of the metric ('views' or 'clones')
        output_dir: Path to directory where CSV files will be saved
    """
    if not data or metric_name not in data:
        print(f"Warning: No {metric_name} data available")
        return
    
    timestamp = datetime.utcnow().isoformat()
    filename = output_dir / f"{metric_name}.csv"
    
    # Prepare new data
    new_rows = []
    for entry in data[metric_name]:
        new_rows.append({
            'fetch_timestamp': timestamp,
            'date': entry['timestamp'][:10],
            'count': entry['count'],
            'uniques': entry['uniques']
        })
    
    new_df = pd.DataFrame(new_rows)
    
    if filename.exists():
        # Load existing data
        existing_df = pd.read_csv(filename)
        
        # Combine old and new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Deduplicate: keep only the most recent fetch_timestamp for each date
        combined_df['fetch_timestamp'] = pd.to_datetime(combined_df['fetch_timestamp'], format='mixed')
        combined_df = combined_df.sort_values('fetch_timestamp').groupby('date').tail(1).reset_index(drop=True)
        
        # Sort by date for clean output
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        combined_df.to_csv(filename, index=False)
    else:
        # Sort new data by date
        new_df = new_df.sort_values('date').reset_index(drop=True)
        new_df.to_csv(filename, index=False)
    
    print(f"Saved {metric_name} data to {filename}")


def save_referrers(data: list, output_dir: Path):
    """Save top referrers (traffic sources) to CSV.
    
    Args:
        data: List of referrer dictionaries from GitHub API
        output_dir: Path to directory where CSV file will be saved
    """
    if not data:
        print("Warning: No referrers data available")
        return
    
    timestamp = datetime.utcnow().isoformat()
    filename = output_dir / "referrers.csv"
    
    # Prepare new data
    new_rows = []
    for ref in data:
        new_rows.append({
            'fetch_timestamp': timestamp,
            'referrer': ref['referrer'],
            'count': ref['count'],
            'uniques': ref['uniques']
        })
    
    new_df = pd.DataFrame(new_rows)
    
    if filename.exists():
        # Load existing data
        existing_df = pd.read_csv(filename)
        
        # Combine old and new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Deduplicate: keep only the most recent fetch_timestamp for each referrer
        combined_df['fetch_timestamp'] = pd.to_datetime(combined_df['fetch_timestamp'], format='mixed')
        combined_df = combined_df.sort_values('fetch_timestamp').groupby('referrer').tail(1).reset_index(drop=True)
        
        # Sort by count descending for easy reading
        combined_df = combined_df.sort_values('count', ascending=False).reset_index(drop=True)
        
        combined_df.to_csv(filename, index=False)
    else:
        # Sort new data by count
        new_df = new_df.sort_values('count', ascending=False).reset_index(drop=True)
        new_df.to_csv(filename, index=False)
    
    print(f"Saved referrers data to {filename}")


def save_popular_paths(data: list, output_dir: Path):
    """Save popular content paths (most viewed pages) to CSV.
    
    Args:
        data: List of path dictionaries from GitHub API
        output_dir: Path to directory where CSV file will be saved
    """
    if not data:
        print("Warning: No popular paths data available")
        return
    
    timestamp = datetime.utcnow().isoformat()
    filename = output_dir / "popular_paths.csv"
    
    # Prepare new data
    new_rows = []
    for path in data:
        new_rows.append({
            'fetch_timestamp': timestamp,
            'path': path['path'],
            'title': path['title'],
            'count': path['count'],
            'uniques': path['uniques']
        })
    
    new_df = pd.DataFrame(new_rows)
    
    if filename.exists():
        # Load existing data
        existing_df = pd.read_csv(filename)
        
        # Combine old and new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Deduplicate: keep only the most recent fetch_timestamp for each path
        combined_df['fetch_timestamp'] = pd.to_datetime(combined_df['fetch_timestamp'], format='mixed')
        combined_df = combined_df.sort_values('fetch_timestamp').groupby('path').tail(1).reset_index(drop=True)
        
        # Sort by count descending for easy reading
        combined_df = combined_df.sort_values('count', ascending=False).reset_index(drop=True)
        
        combined_df.to_csv(filename, index=False)
    else:
        # Sort new data by count
        new_df = new_df.sort_values('count', ascending=False).reset_index(drop=True)
        new_df.to_csv(filename, index=False)
    
    print(f"Saved popular paths data to {filename}")


def save_summary(views: dict, clones: dict, output_dir: Path):
    """Save aggregated weekly summary statistics to CSV.
    
    Args:
        views: Views data dictionary from GitHub API
        clones: Clones data dictionary from GitHub API
        output_dir: Path to directory where CSV file will be saved
    """
    timestamp = datetime.utcnow()
    filename = output_dir / "weekly_summary.csv"
    
    # Calculate ISO week (year-week format like 2026-W04)
    iso_year, iso_week, _ = timestamp.isocalendar()
    week_id = f"{iso_year}-W{iso_week:02d}"
    
    new_row = {
        'week_id': week_id,
        'fetch_timestamp': timestamp.isoformat(),
        'total_views': views.get('count', 0) if views else 0,
        'unique_visitors': views.get('uniques', 0) if views else 0,
        'total_clones': clones.get('count', 0) if clones else 0,
        'unique_cloners': clones.get('uniques', 0) if clones else 0
    }
    
    new_df = pd.DataFrame([new_row])
    
    if filename.exists():
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Parse timestamps with mixed format support
        combined_df['fetch_timestamp'] = pd.to_datetime(combined_df['fetch_timestamp'], format='mixed')
        
        # Deduplicate: keep only one entry per week (most recent fetch)
        # This allows updating the same week's stats if triggered multiple times
        combined_df = combined_df.sort_values('fetch_timestamp').groupby('week_id').tail(1).reset_index(drop=True)
        
        # Sort by week_id for chronological order
        combined_df = combined_df.sort_values('week_id').reset_index(drop=True)
        
        combined_df.to_csv(filename, index=False)
    else:
        new_df.to_csv(filename, index=False)
    
    print(f"Saved weekly summary to {filename}")


def main():
    # Get environment variables
    # Priority to TRAFFIC_TOKEN (Admin PAT), fallback to GITHUB_TOKEN (default action token)
    token = os.environ.get('TRAFFIC_TOKEN') or os.environ.get('GITHUB_TOKEN')
    repo = os.environ.get('GITHUB_REPOSITORY')
    
    if not token or not repo:
        print("Error: TRAFFIC_TOKEN (or GITHUB_TOKEN) and GITHUB_REPOSITORY must be set")
        exit(1)
    
    print(f"Fetching traffic data for {repo}...")
    
    # Create output directory
    output_dir = Path('traffic/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch all traffic data
    views = get_traffic_data(repo, token, 'views')
    clones = get_traffic_data(repo, token, 'clones')
    referrers = get_traffic_data(repo, token, 'popular/referrers')
    paths = get_traffic_data(repo, token, 'popular/paths')
    
    # Save data
    save_timeseries_data(views, 'views', output_dir)
    save_timeseries_data(clones, 'clones', output_dir)
    
    if isinstance(referrers, list):
        save_referrers(referrers, output_dir)
    
    if isinstance(paths, list):
        save_popular_paths(paths, output_dir)
    
    save_summary(views, clones, output_dir)
    
    print("\nTraffic data collection complete.")
    print(f"Data saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
