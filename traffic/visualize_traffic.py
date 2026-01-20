#!/usr/bin/env python3
"""
Visualize repository traffic data from collected CSV files.

Generates histograms and insights from the traffic data stored
in the traffic-data/ directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import sys


def load_traffic_data(data_dir: Path):
    """Load all traffic CSV files from the data directory.
    
    Args:
        data_dir: Path to directory containing traffic CSV files
        
    Returns:
        Dictionary with keys for each traffic metric, values are pandas DataFrames
    """
    data = {}
    
    files = {
        'views': 'views.csv',
        'clones': 'clones.csv',
        'referrers': 'referrers.csv',
        'popular_paths': 'popular_paths.csv',
        'weekly_summary': 'weekly_summary.csv'
    }
    
    for key, filename in files.items():
        filepath = data_dir / filename
        if filepath.exists():
            try:
                data[key] = pd.read_csv(filepath)
                # Convert timestamps to datetime
                if 'fetch_timestamp' in data[key].columns:
                    data[key]['fetch_timestamp'] = pd.to_datetime(data[key]['fetch_timestamp'])
                if 'date' in data[key].columns:
                    data[key]['date'] = pd.to_datetime(data[key]['date'])
                print(f"Loaded {filename}: {len(data[key])} rows")
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
        else:
            print(f"Warning: {filename} not found")
    
    return data


def plot_views_over_time(data: dict, output_dir: Path):
    """Generate timeline plot of repository views and unique visitors.
    
    Args:
        data: Dictionary containing traffic DataFrames
        output_dir: Path to directory where plot will be saved
    """
    if 'views' not in data or data['views'].empty:
        print("Warning: No views data to plot")
        return
    
    df = data['views'].copy()
    
    # Aggregate by date (in case multiple fetches per day)
    df_agg = df.groupby('date').agg({
        'count': 'max',
        'uniques': 'max'
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Total views
    ax1.bar(df_agg['date'], df_agg['count'], color='#2196F3', alpha=0.7)
    ax1.set_title('Total Repository Views Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Views', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.tick_params(axis='x', rotation=45)
    
    # Unique visitors
    ax2.bar(df_agg['date'], df_agg['uniques'], color='#4CAF50', alpha=0.7)
    ax2.set_title('Unique Visitors Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Unique Visitors', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_file = output_dir / 'views_timeline.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved views timeline: {output_file}")
    plt.close()


def plot_clones_over_time(data: dict, output_dir: Path):
    """Generate timeline plot of repository clones and unique cloners.
    
    Args:
        data: Dictionary containing traffic DataFrames
        output_dir: Path to directory where plot will be saved
    """
    if 'clones' not in data or data['clones'].empty:
        print("Warning: No clones data to plot")
        return
    
    df = data['clones'].copy()
    df_agg = df.groupby('date').agg({
        'count': 'max',
        'uniques': 'max'
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Total clones
    ax1.bar(df_agg['date'], df_agg['count'], color='#FF9800', alpha=0.7)
    ax1.set_title('Total Repository Clones Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Clones', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.tick_params(axis='x', rotation=45)
    
    # Unique cloners
    ax2.bar(df_agg['date'], df_agg['uniques'], color='#9C27B0', alpha=0.7)
    ax2.set_title('Unique Cloners Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Unique Cloners', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_file = output_dir / 'clones_timeline.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved clones timeline: {output_file}")
    plt.close()


def plot_weekly_summary(data: dict, output_dir: Path):
    """Generate bar chart comparing views and clones across collection periods.
    
    Args:
        data: Dictionary containing traffic DataFrames
        output_dir: Path to directory where plot will be saved
    """
    if 'weekly_summary' not in data or data['weekly_summary'].empty:
        print("Warning: No weekly summary data to plot")
        return
    
    df = data['weekly_summary'].copy()
    df = df.sort_values('fetch_timestamp')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], df['total_views'], width, 
           label='Total Views', color='#2196F3', alpha=0.7)
    ax.bar([i + width/2 for i in x], df['total_clones'], width,
           label='Total Clones', color='#FF9800', alpha=0.7)
    
    ax.set_title('Weekly Traffic Summary', fontsize=14, fontweight='bold')
    ax.set_xlabel('Collection Date', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df['fetch_timestamp'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'weekly_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved weekly summary: {output_file}")
    plt.close()


def plot_top_referrers(data: dict, output_dir: Path, top_n: int = 10):
    """Generate horizontal bar chart of top traffic referral sources.
    
    Args:
        data: Dictionary containing traffic DataFrames
        output_dir: Path to directory where plot will be saved
        top_n: Number of top referrers to display (default: 10)
    """
    if 'referrers' not in data or data['referrers'].empty:
        print("Warning: No referrers data to plot")
        return
    
    df = data['referrers'].copy()
    
    # Get most recent data
    latest_fetch = df['fetch_timestamp'].max()
    df_latest = df[df['fetch_timestamp'] == latest_fetch]
    
    # Get top N referrers
    df_top = df_latest.nlargest(top_n, 'count')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.barh(range(len(df_top)), df_top['count'], color='#E91E63', alpha=0.7)
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top['referrer'])
    ax.set_xlabel('Number of Views', fontsize=12)
    ax.set_title(f'Top {top_n} Referrers (Latest Data)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {int(width)}',
                ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / 'top_referrers.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved top referrers: {output_file}")
    plt.close()


def plot_popular_paths(data: dict, output_dir: Path, top_n: int = 10):
    """Generate horizontal bar chart of most viewed repository pages.
    
    Args:
        data: Dictionary containing traffic DataFrames
        output_dir: Path to directory where plot will be saved
        top_n: Number of top paths to display (default: 10)
    """
    if 'popular_paths' not in data or data['popular_paths'].empty:
        print("Warning: No popular paths data to plot")
        return
    
    df = data['popular_paths'].copy()
    
    # Get most recent data
    latest_fetch = df['fetch_timestamp'].max()
    df_latest = df[df['fetch_timestamp'] == latest_fetch]
    
    # Get top N paths
    df_top = df_latest.nlargest(top_n, 'count')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(df_top)), df_top['count'], color='#00BCD4', alpha=0.7)
    ax.set_yticks(range(len(df_top)))
    # Truncate long paths for display
    labels = [path[:50] + '...' if len(path) > 50 else path for path in df_top['path']]
    ax.set_yticklabels(labels)
    ax.set_xlabel('Number of Views', fontsize=12)
    ax.set_title(f'Top {top_n} Most Viewed Pages (Latest Data)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {int(width)}',
                ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / 'popular_paths.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved popular paths: {output_file}")
    plt.close()


def generate_insights(data: dict):
    """Generate and print textual insights from collected traffic data.
    
    Args:
        data: Dictionary containing traffic DataFrames
    """
    print("\n" + "="*60)
    print("TRAFFIC INSIGHTS")
    print("="*60)
    
    if 'weekly_summary' in data and not data['weekly_summary'].empty:
        df = data['weekly_summary']
        latest = df.iloc[-1]
        
        print(f"\nLatest Collection: {latest['fetch_timestamp'].strftime('%Y-%m-%d')}")
        print(f"  Total Views: {latest['total_views']:.0f}")
        print(f"  Unique Visitors: {latest['unique_visitors']:.0f}")
        print(f"  Total Clones: {latest['total_clones']:.0f}")
        print(f"  Unique Cloners: {latest['unique_cloners']:.0f}")
        
        if len(df) > 1:
            prev = df.iloc[-2]
            views_change = ((latest['total_views'] - prev['total_views']) / prev['total_views'] * 100)
            print(f"\nWeek-over-week change:")
            print(f"  Views: {views_change:+.1f}%")
    
    if 'referrers' in data and not data['referrers'].empty:
        df = data['referrers']
        latest_fetch = df['fetch_timestamp'].max()
        df_latest = df[df['fetch_timestamp'] == latest_fetch]
        top_ref = df_latest.nlargest(1, 'count').iloc[0]
        print(f"\nTop Referrer: {top_ref['referrer']} ({top_ref['count']:.0f} views)")
    
    if 'popular_paths' in data and not data['popular_paths'].empty:
        df = data['popular_paths']
        latest_fetch = df['fetch_timestamp'].max()
        df_latest = df[df['fetch_timestamp'] == latest_fetch]
        top_path = df_latest.nlargest(1, 'count').iloc[0]
        print(f"\nMost Popular Page: {top_path['path']} ({top_path['count']:.0f} views)")
    
    print("\n" + "="*60)


def main():
    data_dir = Path('traffic/data')
    
    if not data_dir.exists():
        print(f"Error: {data_dir} directory not found")
        print("Run traffic/fetch_traffic.py first to collect data.")
        sys.exit(1)
    
    print("Loading traffic data...")
    data = load_traffic_data(data_dir)
    
    if not data:
        print("Error: No data available to visualize")
        sys.exit(1)
    
    # Create output directory for plots
    output_dir = Path('traffic/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    
    # Generate all plots
    plot_views_over_time(data, output_dir)
    plot_clones_over_time(data, output_dir)
    plot_weekly_summary(data, output_dir)
    plot_top_referrers(data, output_dir)
    plot_popular_paths(data, output_dir)
    
    # Generate insights
    generate_insights(data)
    
    print(f"\nAll visualizations saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
