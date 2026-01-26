#!/usr/bin/env python3
"""
Visualize repository traffic data from collected CSV files.

Generates line graphs and insights from the traffic data stored
in the traffic/data/ directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import sys
import argparse


def parse_arguments():
    """Parse command line arguments for date range filtering.
    
    Returns:
        Tuple of (start_date, end_date, output_dir)
    """
    parser = argparse.ArgumentParser(
        description='Visualize GitHub repository traffic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                              # All data
  %(prog)s 01/01/2026 01/06/2026        # January to June 2026
  %(prog)s 15/03/2026 20/03/2026        # Specific week
        '''
    )
    
    parser.add_argument(
        'start_date',
        nargs='?',
        help='Start date in DD/MM/YYYY format'
    )
    
    parser.add_argument(
        'end_date',
        nargs='?',
        help='End date in DD/MM/YYYY format'
    )
    
    parser.add_argument(
        '--output-dir',
        default='docs/_static/traffic',
        help='Output directory for generated images (default: docs/_static/traffic)'
    )
    
    parser.add_argument(
        '--source',
        default='local',
        help='Data source: "local" for traffic/data/ or "github:owner/repo:branch" for GitHub raw URLs (default: local)'
    )
    
    args = parser.parse_args()
    
    start_date = None
    end_date = None
    
    if args.start_date and args.end_date:
        try:
            start_date = datetime.strptime(args.start_date, '%d/%m/%Y')
            end_date = datetime.strptime(args.end_date, '%d/%m/%Y')
            
            if start_date > end_date:
                print("Error: Start date must be before end date")
                sys.exit(1)
                
            print(f"Filtering data from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
        except ValueError:
            print("Error: Invalid date format. Use DD/MM/YYYY (e.g., 01/01/2026)")
            sys.exit(1)
    elif args.start_date or args.end_date:
        print("Error: Both start and end dates must be provided")
        sys.exit(1)
    
    # Parse data source
    if args.source == 'local':
        data_source = Path('traffic/data')
    elif args.source.startswith('github:'):
        # Format: github:owner/repo:branch
        parts = args.source.replace('github:', '').split(':')
        if len(parts) != 2:
            print("Error: GitHub source format must be 'github:owner/repo:branch'")
            sys.exit(1)
        data_source = tuple(parts)  # (repo, branch)
    else:
        print(f"Error: Unknown source '{args.source}'. Use 'local' or 'github:owner/repo:branch'")
        sys.exit(1)
    
    return start_date, end_date, Path(args.output_dir), data_source


def load_traffic_data(data_source, start_date=None, end_date=None):
    """Load all traffic CSV files from local directory or GitHub raw URLs.
    
    Args:
        data_source: Either Path to local directory or tuple (repo, branch) for GitHub
        start_date: Optional start date for filtering (datetime object)
        end_date: Optional end date for filtering (datetime object)
        
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
    
    # Determine if we're loading from local or GitHub
    if isinstance(data_source, tuple):
        # GitHub URL format
        repo, branch = data_source
        base_url = f"https://raw.githubusercontent.com/{repo}/{branch}/traffic/data"
        print(f"Loading data from GitHub: {repo} (branch: {branch})")
        is_remote = True
    else:
        # Local path
        data_dir = Path(data_source)
        is_remote = False
        print(f"Loading data from local: {data_dir}")
    
    for key, filename in files.items():
        try:
            if is_remote:
                url = f"{base_url}/{filename}"
                df = pd.read_csv(url)
                print(f"Loaded {filename} from GitHub: {len(df)} rows")
            else:
                filepath = data_dir / filename
                if filepath.exists():
                    df = pd.read_csv(filepath)
                    print(f"Loaded {filename}: {len(df)} rows")
                else:
                    print(f"Warning: {filename} not found")
                    continue
            
            # Convert timestamps to datetime
            if 'fetch_timestamp' in df.columns:
                df['fetch_timestamp'] = pd.to_datetime(df['fetch_timestamp'])
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range if specified
            if start_date and end_date:
                if 'date' in df.columns:
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                elif 'fetch_timestamp' in df.columns:
                    df = df[(df['fetch_timestamp'] >= start_date) & (df['fetch_timestamp'] <= end_date)]
            
            if not df.empty:
                data[key] = df
            else:
                print(f"Warning: No data in date range for {filename}")
                
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
    
    return data


def configure_date_axis(ax, dates):
    """Configure x-axis date formatting based on data range.
    
    Args:
        ax: Matplotlib axis object
        dates: Series of datetime objects
    """
    date_range = (dates.max() - dates.min()).days
    
    if date_range <= 31:
        # Less than a month: show day/month
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, date_range // 10)))
    elif date_range <= 180:
        # Less than 6 months: show day/month with fewer labels
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=max(1, date_range // 60)))
    else:
        # More than 6 months: show month/year
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, date_range // 365)))
    
    ax.tick_params(axis='x', rotation=45)


def plot_views_over_time(data: dict, output_dir: Path):
    """Generate line graph of repository views and unique visitors.
    
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
    }).reset_index().sort_values('date')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Total views
    ax1.plot(df_agg['date'], df_agg['count'], color='#2196F3', linewidth=2, marker='o', markersize=4)
    ax1.set_title('Total Repository Views Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Views', fontsize=12)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    configure_date_axis(ax1, df_agg['date'])
    
    # Unique visitors
    ax2.plot(df_agg['date'], df_agg['uniques'], color='#4CAF50', linewidth=2, marker='o', markersize=4)
    ax2.set_title('Unique Visitors Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Unique Visitors', fontsize=12)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    configure_date_axis(ax2, df_agg['date'])
    
    plt.tight_layout()
    output_file = output_dir / 'views_timeline.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved views timeline: {output_file}")
    plt.close()


def plot_clones_over_time(data: dict, output_dir: Path):
    """Generate line graph of repository clones and unique cloners.
    
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
    }).reset_index().sort_values('date')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Total clones
    ax1.plot(df_agg['date'], df_agg['count'], color='#FF9800', linewidth=2, marker='o', markersize=4)
    ax1.set_title('Total Repository Clones Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Clones', fontsize=12)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    configure_date_axis(ax1, df_agg['date'])
    
    # Unique cloners
    ax2.plot(df_agg['date'], df_agg['uniques'], color='#9C27B0', linewidth=2, marker='o', markersize=4)
    ax2.set_title('Unique Cloners Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Unique Cloners', fontsize=12)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    configure_date_axis(ax2, df_agg['date'])
    
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
    ax.set_ylim(bottom=0)
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
    output_file = output_dir / 'top_referrers.png'
    
    if 'referrers' not in data or data['referrers'].empty:
        print("Warning: No referrers data to plot - creating placeholder")
        # Create placeholder image
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No referrer data available yet\n\nReferrer data will appear here once\nthe repository receives external traffic', 
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved placeholder: {output_file}")
        plt.close()
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
    output_file = output_dir / 'popular_paths.png'
    
    if 'popular_paths' not in data or data['popular_paths'].empty:
        print("Warning: No popular paths data to plot - creating placeholder")
        # Create placeholder image
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No popular paths data available yet\n\nPopular content data will appear here once\nthe repository receives traffic', 
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved placeholder: {output_file}")
        plt.close()
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
            if prev['total_views'] > 0:
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
    start_date, end_date, output_dir, data_source = parse_arguments()
    
    # Validate local source exists (skip check for GitHub)
    if isinstance(data_source, Path) and not data_source.exists():
        print(f"Error: {data_source} directory not found")
        print("Run traffic/fetch_traffic.py first to collect data, or use --source github:owner/repo:branch")
        sys.exit(1)
    
    print("Loading traffic data...")
    data = load_traffic_data(data_source, start_date, end_date)
    
    if not data:
        print("Error: No data available to visualize in the specified date range")
        sys.exit(1)
    
    # Create output directory for plots
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
