"""
Exploratory Data Analysis Module
Handles comprehensive EDA, visualizations, and business insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    """
    Perform comprehensive exploratory data analysis
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        pd.DataFrame: Dataset with additional insights
    """
    print("\n" + "="*50)
    print("1. EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # 1.1 Trip Duration Distribution
    print("\n1.1 Trip Duration Distribution")
    print("-" * 30)
    
    if 'trip_duration_minutes' in df.columns:
        duration_stats = df['trip_duration_minutes'].describe()
        print("Trip Duration Statistics:")
        print(duration_stats)
        print(f"\nMedian trip duration: {df['trip_duration_minutes'].median():.2f} minutes")
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(df['trip_duration_minutes'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(df['trip_duration_minutes'].median(), color='red', linestyle='--', 
                   label=f'Median: {df["trip_duration_minutes"].median():.1f} min')
        plt.xlabel('Trip Duration (minutes)')
        plt.ylabel('Frequency')
        plt.title('Trip Duration Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.boxplot(df['trip_duration_minutes'])
        plt.ylabel('Trip Duration (minutes)')
        plt.title('Trip Duration Box Plot')
        
        plt.tight_layout()
        plt.savefig('outputs/trip_duration_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 1.2 Revenue Patterns
    print("\n1.2 Revenue Patterns")
    print("-" * 20)
    
    if 'pickup_hour' in df.columns and 'total_amount' in df.columns:
        hourly_revenue = df.groupby('pickup_hour')['total_amount'].agg(['mean', 'count', 'sum']).round(2)
        hourly_revenue.columns = ['Avg_Fare', 'Trip_Count', 'Total_Revenue']
        
        peak_hour = hourly_revenue['Avg_Fare'].idxmax()
        peak_fare = hourly_revenue['Avg_Fare'].max()
        
        print(f"Peak revenue hour: {peak_hour}:00 with average fare of ${peak_fare}")
        print("\nHourly Revenue Summary:")
        print(hourly_revenue)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.bar(hourly_revenue.index, hourly_revenue['Avg_Fare'], color='lightcoral')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Fare ($)')
        plt.title('Average Fare by Hour')
        plt.xticks(range(0, 24, 2))
        
        plt.subplot(1, 3, 2)
        plt.bar(hourly_revenue.index, hourly_revenue['Trip_Count'], color='lightgreen')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Trips')
        plt.title('Trip Count by Hour')
        plt.xticks(range(0, 24, 2))
        
        plt.subplot(1, 3, 3)
        plt.bar(hourly_revenue.index, hourly_revenue['Total_Revenue'], color='lightblue')
        plt.xlabel('Hour of Day')
        plt.ylabel('Total Revenue ($)')
        plt.title('Total Revenue by Hour')
        plt.xticks(range(0, 24, 2))
        
        plt.tight_layout()
        plt.savefig('outputs/revenue_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 1.3 Pickup Distribution
    print("\n1.3 Pickup Distribution")
    print("-" * 22)
    
    pickup_cols = ['PULocationID', 'pickup_zone'] if 'pickup_zone' in df.columns else ['PULocationID']
    
    for col in pickup_cols:
        if col in df.columns:
            pickup_dist = df[col].value_counts().head(10)
            print(f"\nTop 10 Pickup Locations ({col}):")
            print(pickup_dist)
            
            plt.figure(figsize=(12, 6))
            pickup_dist.plot(kind='bar', color='orange')
            plt.title(f'Top 10 Pickup Locations - {col}')
            plt.xlabel('Location')
            plt.ylabel('Number of Trips')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('outputs/pickup_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            break
    
    # 1.4 Passenger-Tip Relationship
    print("\n1.4 Passenger-Tip Relationship")
    print("-" * 30)
    
    if 'passenger_count' in df.columns and 'tip_amount' in df.columns:
        # Filter reasonable passenger counts
        df_tips = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]
        
        tip_analysis = df_tips.groupby('passenger_count')['tip_amount'].agg(['mean', 'median', 'count']).round(2)
        tip_analysis.columns = ['Avg_Tip', 'Median_Tip', 'Trip_Count']
        
        print("Tip Analysis by Passenger Count:")
        print(tip_analysis)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.bar(tip_analysis.index, tip_analysis['Avg_Tip'], color='gold')
        plt.xlabel('Passenger Count')
        plt.ylabel('Average Tip ($)')
        plt.title('Average Tip by Passenger Count')
        
        plt.subplot(1, 3, 2)
        df_tips.boxplot(column='tip_amount', by='passenger_count', ax=plt.gca())
        plt.title('Tip Distribution by Passenger Count')
        plt.suptitle('')
        
        plt.subplot(1, 3, 3)
        plt.scatter(df_tips['passenger_count'], df_tips['tip_amount'], alpha=0.1)
        plt.xlabel('Passenger Count')
        plt.ylabel('Tip Amount ($)')
        plt.title('Passenger Count vs Tip Amount')
        
        plt.tight_layout()
        plt.savefig('outputs/passenger_tip_relationship.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation
        correlation = df_tips['passenger_count'].corr(df_tips['tip_amount'])
        print(f"\nCorrelation between passenger count and tip amount: {correlation:.3f}")
    
    # 1.5 Trip Records Summary - Longest Trips
    print("\n1.5 Longest Trips Summary")
    print("-" * 25)
    
    if 'trip_duration_minutes' in df.columns:
        longest_trips = df.nlargest(10, 'trip_duration_minutes')
        
        display_cols = ['trip_duration_minutes', 'trip_distance', 'total_amount', 
                       'pickup_hour', 'PULocationID', 'DOLocationID']
        display_cols = [col for col in display_cols if col in longest_trips.columns]
        
        print("Top 10 Longest Trips:")
        print(longest_trips[display_cols].round(2))
        
        # Summary statistics for longest trips
        print(f"\nLongest trip duration: {longest_trips['trip_duration_minutes'].iloc[0]:.1f} minutes")
        print(f"Average duration of top 10 longest trips: {longest_trips['trip_duration_minutes'].mean():.1f} minutes")
        if 'trip_distance' in longest_trips.columns:
            print(f"Average distance of top 10 longest trips: {longest_trips['trip_distance'].mean():.1f} miles")
    
    # 1.6 Overall Summary
    print("\n1.6 EDA Business Summary")
    print("-" * 25)
    
    summary_insights = []
    
    if 'trip_duration_minutes' in df.columns:
        avg_duration = df['trip_duration_minutes'].mean()
        summary_insights.append(f"• Average trip duration: {avg_duration:.1f} minutes")
    
    if 'pickup_hour' in df.columns and 'total_amount' in df.columns:
        peak_hour = df.groupby('pickup_hour')['total_amount'].mean().idxmax()
        summary_insights.append(f"• Peak revenue hour: {peak_hour}:00")
    
    if 'PULocationID' in df.columns:
        busiest_location = df['PULocationID'].mode().iloc[0]
        summary_insights.append(f"• Busiest pickup location ID: {busiest_location}")
    
    if 'passenger_count' in df.columns:
        avg_passengers = df['passenger_count'].mean()
        summary_insights.append(f"• Average passengers per trip: {avg_passengers:.1f}")
    
    print("Key Business Insights:")
    for insight in summary_insights:
        print(insight)
    
    return df

def advanced_time_series_analysis(df):
    """
    Perform time series analysis on trip patterns
    
    Args:
        df (pd.DataFrame): Cleaned dataset
    """
    if 'tpep_pickup_datetime' not in df.columns:
        return
    
    print("\n" + "="*50)
    print("ADVANCED TIME SERIES ANALYSIS")
    print("="*50)
    
    # Daily patterns
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
    daily_trips = df.groupby('pickup_date').size()
    
    plt.figure(figsize=(15, 10))
    
    # Daily trip count
    plt.subplot(2, 2, 1)
    daily_trips.plot(kind='line')
    plt.title('Daily Trip Count Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Trips')
    plt.xticks(rotation=45)
    
    # Hourly heatmap
    plt.subplot(2, 2, 2)
    df['pickup_day_name'] = df['tpep_pickup_datetime'].dt.day_name()
    hourly_patterns = df.groupby(['pickup_day_name', 'pickup_hour']).size().unstack(fill_value=0)
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hourly_patterns = hourly_patterns.reindex(day_order)
    
    sns.heatmap(hourly_patterns, cmap='YlOrRd', annot=False, fmt='d')
    plt.title('Trip Patterns: Day vs Hour Heatmap')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    
    # Monthly trends
    plt.subplot(2, 2, 3)
    df['pickup_month_name'] = df['tpep_pickup_datetime'].dt.strftime('%B')
    monthly_trips = df.groupby('pickup_month_name').size()
    
    # Order months correctly
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_trips = monthly_trips.reindex([m for m in month_order if m in monthly_trips.index])
    
    monthly_trips.plot(kind='bar', color='skyblue')
    plt.title('Monthly Trip Distribution')
    plt.xlabel('Month')
    plt.ylabel('Number of Trips')
    plt.xticks(rotation=45)
    
    # Weekend vs Weekday
    plt.subplot(2, 2, 4)
    df['is_weekend'] = df['pickup_weekday'].isin([5, 6])
    weekend_comparison = df.groupby('is_weekend').agg({
        'trip_duration_minutes': 'mean',
        'total_amount': 'mean'
    })
    
    weekend_comparison.index = ['Weekday', 'Weekend']
    weekend_comparison.plot(kind='bar', color=['lightcoral', 'lightgreen'])
    plt.title('Weekend vs Weekday Comparison')
    plt.xlabel('Day Type')
    plt.ylabel('Average Value')
    plt.xticks(rotation=0)
    plt.legend(['Duration (min)', 'Fare ($)'])
    
    plt.tight_layout()
    plt.savefig('outputs/time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def payment_analysis(df):
    """
    Analyze payment patterns and methods
    
    Args:
        df (pd.DataFrame): Cleaned dataset
    """
    if 'payment_type' not in df.columns:
        return
    
    print("\n" + "="*50)
    print("PAYMENT METHOD ANALYSIS")
    print("="*50)
    
    # Payment type distribution
    payment_dist = df['payment_type'].value_counts()
    print("Payment Type Distribution:")
    print(payment_dist)
    
    # Payment method mapping (common codes)
    payment_mapping = {
        1: 'Credit Card',
        2: 'Cash',
        3: 'No Charge',
        4: 'Dispute',
        5: 'Unknown',
        6: 'Voided Trip'
    }
    
    df['payment_method'] = df['payment_type'].map(payment_mapping).fillna('Other')
    
    plt.figure(figsize=(15, 5))
    
    # Payment distribution pie chart
    plt.subplot(1, 3, 1)
    payment_counts = df['payment_method'].value_counts()
    plt.pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%')
    plt.title('Payment Method Distribution')
    
    # Tip analysis by payment method
    plt.subplot(1, 3, 2)
    if 'tip_amount' in df.columns:
        tip_by_payment = df.groupby('payment_method')['tip_amount'].mean()
        tip_by_payment.plot(kind='bar', color='gold')
        plt.title('Average Tip by Payment Method')
        plt.xlabel('Payment Method')
        plt.ylabel('Average Tip ($)')
        plt.xticks(rotation=45)
    
    # Fare analysis by payment method
    plt.subplot(1, 3, 3)
    if 'total_amount' in df.columns:
        fare_by_payment = df.groupby('payment_method')['total_amount'].mean()
        fare_by_payment.plot(kind='bar', color='lightcoral')
        plt.title('Average Fare by Payment Method')
        plt.xlabel('Payment Method')
        plt.ylabel('Average Fare ($)')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/payment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_dashboard_visualizations(df):
    """
    Create comprehensive dashboard-style visualizations
    
    Args:
        df (pd.DataFrame): Cleaned dataset
    """
    print("\n" + "="*50)
    print("DASHBOARD VISUALIZATIONS")
    print("="*50)
    
    # Set up the dashboard
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Trip Duration Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if 'trip_duration_minutes' in df.columns:
        df['trip_duration_minutes'].hist(bins=50, alpha=0.7, color='skyblue', ax=ax1)
        ax1.axvline(df['trip_duration_minutes'].median(), color='red', linestyle='--')
        ax1.set_title('Trip Duration Distribution')
        ax1.set_xlabel('Duration (min)')
        ax1.set_ylabel('Frequency')
    
    # 2. Hourly Revenue Pattern
    ax2 = fig.add_subplot(gs[0, 1])
    if 'pickup_hour' in df.columns and 'total_amount' in df.columns:
        hourly_revenue = df.groupby('pickup_hour')['total_amount'].mean()
        hourly_revenue.plot(kind='line', marker='o', color='green', ax=ax2)
        ax2.set_title('Average Fare by Hour')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Average Fare ($)')
    
    # 3. Top Pickup Locations
    ax3 = fig.add_subplot(gs[0, 2])
    if 'PULocationID' in df.columns:
        top_pickups = df['PULocationID'].value_counts().head(10)
        top_pickups.plot(kind='bar', color='orange', ax=ax3)
        ax3.set_title('Top 10 Pickup Locations')
        ax3.set_xlabel('Location ID')
        ax3.set_ylabel('Trip Count')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Payment Method Distribution
    ax4 = fig.add_subplot(gs[0, 3])
    if 'payment_type' in df.columns:
        payment_counts = df['payment_type'].value_counts()
        ax4.pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%')
        ax4.set_title('Payment Methods')
    
    # 5. Distance vs Duration Scatter
    ax5 = fig.add_subplot(gs[1, :2])
    if 'trip_distance' in df.columns and 'trip_duration_minutes' in df.columns:
        sample_data = df.sample(n=min(5000, len(df)))
        ax5.scatter(sample_data['trip_distance'], sample_data['trip_duration_minutes'], 
                   alpha=0.5, color='purple')
        ax5.set_title('Trip Distance vs Duration')
        ax5.set_xlabel('Distance (miles)')
        ax5.set_ylabel('Duration (minutes)')
    
    # 6. Weekly Pattern Heatmap
    ax6 = fig.add_subplot(gs[1, 2:])
    if 'pickup_hour' in df.columns:
        df['day_name'] = df['tpep_pickup_datetime'].dt.day_name() if 'tpep_pickup_datetime' in df.columns else 'Unknown'
        weekly_pattern = df.groupby(['day_name', 'pickup_hour']).size().unstack(fill_value=0)
        
        if not weekly_pattern.empty:
            sns.heatmap(weekly_pattern, cmap='YlOrRd', annot=False, ax=ax6)
            ax6.set_title('Weekly Trip Pattern Heatmap')
            ax6.set_xlabel('Hour')
            ax6.set_ylabel('Day')
    
    # 7. Monthly Trends
    ax7 = fig.add_subplot(gs[2, :2])
    if 'tpep_pickup_datetime' in df.columns:
        df['month'] = df['tpep_pickup_datetime'].dt.month
        monthly_stats = df.groupby('month').agg({
            'trip_duration_minutes': 'mean',
            'total_amount': 'mean'
        })
        
        ax7_twin = ax7.twinx()
        line1 = ax7.plot(monthly_stats.index, monthly_stats['trip_duration_minutes'], 
                        'b-o', label='Avg Duration')
        line2 = ax7_twin.plot(monthly_stats.index, monthly_stats['total_amount'], 
                             'r-s', label='Avg Fare')
        
        ax7.set_xlabel('Month')
        ax7.set_ylabel('Duration (min)', color='b')
        ax7_twin.set_ylabel('Fare ($)', color='r')
        ax7.set_title('Monthly Trends')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax7.legend(lines, labels, loc='upper left')
    
    # 8. Fare Components Breakdown
    ax8 = fig.add_subplot(gs[2, 2:])
    fare_components = ['fare_amount', 'tip_amount', 'tolls_amount', 'extra']
    available_components = [col for col in fare_components if col in df.columns]
    
    if available_components:
        component_means = df[available_components].mean()
        component_means.plot(kind='bar', color=['lightblue', 'gold', 'lightcoral', 'lightgreen'], ax=ax8)
        ax8.set_title('Average Fare Components')
        ax8.set_xlabel('Component')
        ax8.set_ylabel('Average Amount ($)')
        ax8.tick_params(axis='x', rotation=45)
    
    # 9. Performance Metrics Summary
    ax9 = fig.add_subplot(gs[3, :])
    
    # Create summary statistics table
    summary_stats = []
    
    if 'trip_duration_minutes' in df.columns:
        summary_stats.append(['Avg Trip Duration', f"{df['trip_duration_minutes'].mean():.1f} min"])
        summary_stats.append(['Median Trip Duration', f"{df['trip_duration_minutes'].median():.1f} min"])
    
    if 'total_amount' in df.columns:
        summary_stats.append(['Avg Fare', f"${df['total_amount'].mean():.2f}"])
        summary_stats.append(['Total Revenue', f"${df['total_amount'].sum():,.2f}"])
    
    if 'trip_distance' in df.columns:
        summary_stats.append(['Avg Distance', f"{df['trip_distance'].mean():.1f} miles"])
    
    summary_stats.append(['Total Trips', f"{len(df):,}"])
    
    if 'PULocationID' in df.columns:
        summary_stats.append(['Unique Locations', f"{df['PULocationID'].nunique():,}"])
    
    # Display as table
    ax9.axis('tight')
    ax9.axis('off')
    table = ax9.table(cellText=summary_stats,
                     colLabels=['Metric', 'Value'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax9.set_title('Key Performance Indicators', pad=20)
    
    plt.suptitle('NYC Taxi Operations Dashboard', fontsize=16, y=0.98)
    plt.savefig('outputs/dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_report_summary(df):
    """
    Generate executive summary report
    
    Args:
        df (pd.DataFrame): Cleaned dataset
    """
    print("\n" + "="*50)
    print("EXECUTIVE SUMMARY REPORT")
    print("="*50)
    
    # Key Performance Indicators
    total_trips = len(df)
    avg_duration = df['trip_duration_minutes'].mean() if 'trip_duration_minutes' in df.columns else 0
    avg_fare = df['total_amount'].mean() if 'total_amount' in df.columns else 0
    avg_distance = df['trip_distance'].mean() if 'trip_distance' in df.columns else 0
    
    print(f"\nKey Performance Indicators:")
    print(f"• Total trips analyzed: {total_trips:,}")
    print(f"• Average trip duration: {avg_duration:.1f} minutes")
    print(f"• Average fare: ${avg_fare:.2f}")
    print(f"• Average trip distance: {avg_distance:.1f} miles")
    
    # Revenue insights
    if 'pickup_hour' in df.columns and 'total_amount' in df.columns:
        peak_revenue_hour = df.groupby('pickup_hour')['total_amount'].sum().idxmax()
        total_revenue = df['total_amount'].sum()
        print(f"• Peak revenue hour: {peak_revenue_hour}:00")
        print(f"• Total revenue analyzed: ${total_revenue:,.2f}")
    
    # Operational insights
    if 'PULocationID' in df.columns:
        unique_pickup_locations = df['PULocationID'].nunique()
        busiest_location = df['PULocationID'].mode().iloc[0]
        print(f"• Unique pickup locations: {unique_pickup_locations}")
        print(f"• Busiest pickup location: {busiest_location}")
    
    print("\nStrategic Recommendations:")
    print("• Optimize fleet allocation during peak hours for maximum revenue")
    print("• Focus driver recruitment in high-demand pickup zones")
    print("• Implement dynamic pricing based on duration predictions")
    print("• Use route optimization for corporate delivery services") 