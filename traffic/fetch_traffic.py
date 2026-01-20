#!/usr/bin/env python3
"""
Fetch GitHub repository traffic data and store it as CSV.

This script collects:
- Views (total and unique)
- Clones (total and unique)
- Top referrers
- Popular content (paths)

Data is stored in traffic-data/ directory with timestamps.
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
import requests


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
    
    # Check if file exists to determine if we need headers
    file_exists = filename.exists()
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            writer.writerow(['fetch_timestamp', 'date', 'count', 'uniques'])
        
        # Write data points
        for entry in data[metric_name]:
            writer.writerow([
                timestamp,
                entry['timestamp'][:10],  # Just the date part
                entry['count'],
                entry['uniques']
            ])
    
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
    
    file_exists = filename.exists()
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['fetch_timestamp', 'referrer', 'count', 'uniques'])
        
        for ref in data:
            writer.writerow([
                timestamp,
                ref['referrer'],
                ref['count'],
                ref['uniques']
            ])
    
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
    
    file_exists = filename.exists()
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['fetch_timestamp', 'path', 'title', 'count', 'uniques'])
        
        for path in data:
            writer.writerow([
                timestamp,
                path['path'],
                path['title'],
                path['count'],
                path['uniques']
            ])
    
    print(f"Saved popular paths data to {filename}")


def save_summary(views: dict, clones: dict, output_dir: Path):
    """Save aggregated weekly summary statistics to CSV.
    
    Args:
        views: Views data dictionary from GitHub API
        clones: Clones data dictionary from GitHub API
        output_dir: Path to directory where CSV file will be saved
    """
    timestamp = datetime.utcnow().isoformat()
    filename = output_dir / "weekly_summary.csv"
    
    file_exists = filename.exists()
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'fetch_timestamp',
                'total_views',
                'unique_visitors',
                'total_clones',
                'unique_cloners'
            ])
        
        writer.writerow([
            timestamp,
            views.get('count', 0) if views else 0,
            views.get('uniques', 0) if views else 0,
            clones.get('count', 0) if clones else 0,
            clones.get('uniques', 0) if clones else 0
        ])
    
    print(f"Saved weekly summary to {filename}")


def main():
    # Get environment variables
    token = os.environ.get('GITHUB_TOKEN')
    repo = os.environ.get('GITHUB_REPOSITORY')
    
    if not token or not repo:
        print("Error: GITHUB_TOKEN and GITHUB_REPOSITORY must be set")
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
