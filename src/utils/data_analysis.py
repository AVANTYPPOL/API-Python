"""
Enhanced Uber Dataset Analysis
=============================

Comprehensive analysis of the enhanced Uber price dataset to understand
data patterns and provide recommendations for model improvement.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DatasetAnalyzer:
    """Comprehensive analysis of the enhanced Uber dataset"""
    
    def __init__(self, db_path='uber_ml_data.db'):
        self.db_path = db_path
        self.df = None
        
    def load_data(self):
        """Load data from SQLite database"""
        print("📥 Loading Dataset for Analysis...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load all data (including incomplete records)
        query = "SELECT * FROM rides WHERE uber_price_usd > 0"
        self.df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"✅ Loaded {len(self.df)} total records")
        return self.df
    
    def basic_statistics(self):
        """Display basic dataset statistics"""
        print("\n" + "="*60)
        print("📊 DATASET OVERVIEW")
        print("="*60)
        
        # Dataset size
        print(f"Total Records: {len(self.df):,}")
        print(f"Total Features: {len(self.df.columns)}")
        
        # Target variable stats
        price_stats = self.df['uber_price_usd'].describe()
        print(f"\n💰 Price Statistics:")
        print(f"   Mean: ${price_stats['mean']:.2f}")
        print(f"   Median: ${price_stats['50%']:.2f}")
        print(f"   Min: ${price_stats['min']:.2f}")
        print(f"   Max: ${price_stats['max']:.2f}")
        print(f"   Std Dev: ${price_stats['std']:.2f}")
        
        # Distance stats
        dist_stats = self.df['distance_km'].describe()
        print(f"\n📏 Distance Statistics:")
        print(f"   Mean: {dist_stats['mean']:.2f} km")
        print(f"   Median: {dist_stats['50%']:.2f} km")
        print(f"   Min: {dist_stats['min']:.2f} km")
        print(f"   Max: {dist_stats['max']:.2f} km")
        
        # Data completeness
        print(f"\n🎯 Data Completeness:")
        complete_records = self.df.dropna().shape[0]
        print(f"   Complete records: {complete_records}/{len(self.df)} ({complete_records/len(self.df)*100:.1f}%)")
        
        # Traffic/Weather data availability
        traffic_available = self.df['traffic_level'].notna().sum()
        weather_available = self.df['weather_condition'].notna().sum()
        print(f"   With traffic data: {traffic_available}/{len(self.df)} ({traffic_available/len(self.df)*100:.1f}%)")
        print(f"   With weather data: {weather_available}/{len(self.df)} ({weather_available/len(self.df)*100:.1f}%)")
    
    def analyze_categorical_features(self):
        """Analyze categorical features distribution"""
        print("\n" + "="*60)
        print("🏷️ CATEGORICAL FEATURES ANALYSIS")
        print("="*60)
        
        categorical_features = ['traffic_level', 'weather_condition']
        
        for feature in categorical_features:
            if feature in self.df.columns:
                print(f"\n{feature.upper()}:")
                
                # Value counts
                value_counts = self.df[feature].value_counts(dropna=False)
                total = len(self.df)
                
                for value, count in value_counts.items():
                    pct = (count / total) * 100
                    print(f"   {str(value):12s}: {count:3d} records ({pct:5.1f}%)")
                
                # Price analysis by category
                if not self.df[feature].isna().all():
                    print(f"\n   Average Price by {feature}:")
                    price_by_category = self.df.groupby(feature)['uber_price_usd'].agg(['mean', 'count'])
                    for category, row in price_by_category.iterrows():
                        print(f"   {str(category):12s}: ${row['mean']:6.2f} (n={row['count']})")
    
    def analyze_time_patterns(self):
        """Analyze temporal patterns in the data"""
        print("\n" + "="*60)
        print("⏰ TEMPORAL PATTERNS ANALYSIS")
        print("="*60)
        
        # Hour of day analysis
        print("\nPRICE BY HOUR OF DAY:")
        hour_analysis = self.df.groupby('hour_of_day')['uber_price_usd'].agg(['mean', 'count'])
        for hour, row in hour_analysis.iterrows():
            print(f"   {hour:2d}:00: ${row['mean']:6.2f} (n={row['count']})")
        
        # Day of week analysis
        print("\nPRICE BY DAY OF WEEK:")
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_analysis = self.df.groupby('day_of_week')['uber_price_usd'].agg(['mean', 'count'])
        for day, row in day_analysis.iterrows():
            day_name = day_names[day] if day < len(day_names) else f"Day_{day}"
            print(f"   {day_name:9s}: ${row['mean']:6.2f} (n={row['count']})")
        
        # Surge multiplier analysis
        print(f"\nSURGE MULTIPLIER STATS:")
        surge_stats = self.df['surge_multiplier'].describe()
        print(f"   Mean: {surge_stats['mean']:.2f}")
        print(f"   Min: {surge_stats['min']:.2f}")
        print(f"   Max: {surge_stats['max']:.2f}")
        
        surge_distribution = self.df['surge_multiplier'].value_counts().sort_index()
        print(f"\n   Surge Distribution:")
        for surge, count in surge_distribution.items():
            pct = (count / len(self.df)) * 100
            print(f"   {surge:.1f}x: {count:3d} records ({pct:5.1f}%)")
    
    def analyze_location_patterns(self):
        """Analyze geographic patterns"""
        print("\n" + "="*60)
        print("🗺️ LOCATION PATTERNS ANALYSIS")
        print("="*60)
        
        # Coordinate ranges
        print("COORDINATE RANGES:")
        print(f"   Pickup Latitude:  {self.df['pickup_lat'].min():.4f} to {self.df['pickup_lat'].max():.4f}")
        print(f"   Pickup Longitude: {self.df['pickup_lng'].min():.4f} to {self.df['pickup_lng'].max():.4f}")
        print(f"   Dropoff Latitude: {self.df['dropoff_lat'].min():.4f} to {self.df['dropoff_lat'].max():.4f}")
        print(f"   Dropoff Longitude:{self.df['dropoff_lng'].min():.4f} to {self.df['dropoff_lng'].max():.4f}")
        
        # Calculate distances from downtown Miami
        downtown_lat, downtown_lng = 25.7617, -80.1918
        
        pickup_dist = np.sqrt(
            (self.df['pickup_lat'] - downtown_lat) ** 2 + 
            (self.df['pickup_lng'] - downtown_lng) ** 2
        )
        
        dropoff_dist = np.sqrt(
            (self.df['dropoff_lat'] - downtown_lat) ** 2 + 
            (self.df['dropoff_lng'] - downtown_lng) ** 2
        )
        
        print(f"\nDISTANCE FROM DOWNTOWN MIAMI:")
        print(f"   Avg Pickup Distance:  {pickup_dist.mean():.4f} degrees")
        print(f"   Avg Dropoff Distance: {dropoff_dist.mean():.4f} degrees")
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        print("\n" + "="*60)
        print("🔗 CORRELATION ANALYSIS")
        print("="*60)
        
        # Select numerical features
        numerical_features = [
            'distance_km', 'pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng',
            'hour_of_day', 'day_of_week', 'surge_multiplier', 'uber_price_usd'
        ]
        
        correlation_matrix = self.df[numerical_features].corr()
        
        # Show correlations with price
        price_correlations = correlation_matrix['uber_price_usd'].abs().sort_values(ascending=False)
        
        print("CORRELATIONS WITH PRICE (absolute values):")
        for feature, corr in price_correlations.items():
            if feature != 'uber_price_usd':  # Exclude self-correlation
                print(f"   {feature:20s}: {corr:.4f}")
        
        # Find strongest correlations
        print(f"\nSTRONGEST PRICE PREDICTORS:")
        top_correlations = price_correlations.drop('uber_price_usd').head(3)
        for feature, corr in top_correlations.items():
            print(f"   {feature}: {corr:.4f}")
    
    def data_quality_assessment(self):
        """Assess data quality issues"""
        print("\n" + "="*60)
        print("🔍 DATA QUALITY ASSESSMENT")
        print("="*60)
        
        # Check for outliers
        print("OUTLIER DETECTION:")
        
        # Price outliers (using IQR method)
        Q1 = self.df['uber_price_usd'].quantile(0.25)
        Q3 = self.df['uber_price_usd'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        price_outliers = self.df[
            (self.df['uber_price_usd'] < lower_bound) | 
            (self.df['uber_price_usd'] > upper_bound)
        ]
        
        print(f"   Price outliers: {len(price_outliers)} records")
        if len(price_outliers) > 0:
            print(f"   Outlier prices: ${price_outliers['uber_price_usd'].min():.2f} - ${price_outliers['uber_price_usd'].max():.2f}")
        
        # Distance outliers
        Q1_dist = self.df['distance_km'].quantile(0.25)
        Q3_dist = self.df['distance_km'].quantile(0.75)
        IQR_dist = Q3_dist - Q1_dist
        lower_bound_dist = Q1_dist - 1.5 * IQR_dist
        upper_bound_dist = Q3_dist + 1.5 * IQR_dist
        
        distance_outliers = self.df[
            (self.df['distance_km'] < lower_bound_dist) | 
            (self.df['distance_km'] > upper_bound_dist)
        ]
        
        print(f"   Distance outliers: {len(distance_outliers)} records")
        if len(distance_outliers) > 0:
            print(f"   Outlier distances: {distance_outliers['distance_km'].min():.2f} - {distance_outliers['distance_km'].max():.2f} km")
        
        # Check price per km ratio
        price_per_km = self.df['uber_price_usd'] / self.df['distance_km']
        print(f"\nPRICE PER KM ANALYSIS:")
        print(f"   Mean: ${price_per_km.mean():.2f}/km")
        print(f"   Median: ${price_per_km.median():.2f}/km")
        print(f"   Range: ${price_per_km.min():.2f} - ${price_per_km.max():.2f}/km")
    
    def recommendations(self):
        """Provide recommendations for model improvement"""
        print("\n" + "="*60)
        print("🎯 RECOMMENDATIONS FOR MODEL IMPROVEMENT")
        print("="*60)
        
        complete_records = self.df.dropna().shape[0]
        traffic_records = self.df['traffic_level'].notna().sum()
        weather_records = self.df['weather_condition'].notna().sum()
        
        print("1. DATA COLLECTION PRIORITIES:")
        print(f"   • Collect more data! Current: {len(self.df)} records")
        print(f"   • Target: 500+ records for reliable ML model")
        print(f"   • Focus on complete records with traffic/weather data")
        
        print("\n2. FEATURE ENGINEERING OPPORTUNITIES:")
        print("   • Create more location-based features (airport, beach, downtown zones)")
        print("   • Add time-based features (rush hour intensity, weekend effects)")
        print("   • Consider weather severity categories instead of just condition")
        print("   • Add traffic congestion levels (1-10 scale)")
        
        print("\n3. MODEL IMPROVEMENTS:")
        print("   • Use regularization to prevent overfitting (Ridge/Lasso)")
        print("   • Try ensemble methods (XGBoost, LightGBM)")
        print("   • Implement cross-validation for small datasets")
        print("   • Consider polynomial features for distance")
        
        print("\n4. DATA QUALITY ACTIONS:")
        if len(self.df) < 100:
            print("   • CRITICAL: Need more data points for reliable predictions")
        
        if traffic_records < len(self.df) * 0.8:
            print("   • Improve traffic data collection (currently incomplete)")
        
        if weather_records < len(self.df) * 0.8:
            print("   • Improve weather data collection (currently incomplete)")
        
        print("\n5. PRICING INSIGHTS:")
        price_per_km = self.df['uber_price_usd'] / self.df['distance_km']
        print(f"   • Average pricing: ${price_per_km.mean():.2f} per km")
        print(f"   • Use this as baseline for competitive pricing")
        print(f"   • Consider distance-based pricing tiers")
    
    def create_visualizations(self):
        """Create comprehensive data visualizations"""
        print("\n📊 Creating Data Visualizations...")
        
        # Set up the plot style
        plt.rcParams['figure.figsize'] = (15, 12)
        
        # Create a comprehensive dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Uber Dataset Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Price distribution
        axes[0, 0].hist(self.df['uber_price_usd'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Price (USD)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Distance vs Price scatter
        axes[0, 1].scatter(self.df['distance_km'], self.df['uber_price_usd'], alpha=0.6, color='coral')
        axes[0, 1].set_title('Distance vs Price')
        axes[0, 1].set_xlabel('Distance (km)')
        axes[0, 1].set_ylabel('Price (USD)')
        
        # Add trendline
        z = np.polyfit(self.df['distance_km'], self.df['uber_price_usd'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(self.df['distance_km'], p(self.df['distance_km']), "r--", alpha=0.8)
        
        # 3. Price by hour of day
        hour_data = self.df.groupby('hour_of_day')['uber_price_usd'].mean()
        axes[0, 2].bar(hour_data.index, hour_data.values, alpha=0.7, color='lightgreen')
        axes[0, 2].set_title('Average Price by Hour')
        axes[0, 2].set_xlabel('Hour of Day')
        axes[0, 2].set_ylabel('Average Price (USD)')
        
        # 4. Traffic level distribution (if available)
        if self.df['traffic_level'].notna().any():
            traffic_counts = self.df['traffic_level'].value_counts()
            axes[1, 0].pie(traffic_counts.values, labels=traffic_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Traffic Level Distribution')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Traffic Data\nAvailable', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Traffic Level Distribution')
        
        # 5. Weather condition distribution (if available)
        if self.df['weather_condition'].notna().any():
            weather_counts = self.df['weather_condition'].value_counts()
            axes[1, 1].pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Weather Condition Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Weather Data\nAvailable', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Weather Condition Distribution')
        
        # 6. Surge multiplier impact
        surge_data = self.df.groupby('surge_multiplier')['uber_price_usd'].mean()
        axes[1, 2].bar(surge_data.index, surge_data.values, alpha=0.7, color='gold')
        axes[1, 2].set_title('Price by Surge Multiplier')
        axes[1, 2].set_xlabel('Surge Multiplier')
        axes[1, 2].set_ylabel('Average Price (USD)')
        
        plt.tight_layout()
        plt.savefig('dataset_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Dashboard saved as 'dataset_analysis_dashboard.png'")
    
    def run_complete_analysis(self):
        """Run the complete analysis suite"""
        print("🚀 STARTING COMPREHENSIVE DATASET ANALYSIS")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Run all analysis modules
        self.basic_statistics()
        self.analyze_categorical_features()
        self.analyze_time_patterns()
        self.analyze_location_patterns()
        self.correlation_analysis()
        self.data_quality_assessment()
        self.recommendations()
        
        # Create visualizations
        self.create_visualizations()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print("Check 'dataset_analysis_dashboard.png' for visual insights")

def main():
    """Main function to run the analysis"""
    analyzer = DatasetAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 