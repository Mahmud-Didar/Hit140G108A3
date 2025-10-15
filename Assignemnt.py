"""
Investigation A: Do bats perceive rats as potential predators?
Investigation B: Do behaviors change with seasonal changes?
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import logit
from statsmodels.formula.api import ols
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("="*80)
print("BAT VS. RAT: THE FORAGE FILES - COMPREHENSIVE ANALYSIS")
print("="*80)

# ============================================================================
# SECTION 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: DATA LOADING AND INITIAL EXPLORATION")
print("="*80)

# Load datasets
try:
    dataset1 = pd.read_csv("dataset1.csv")
    print(" Successfully loaded dataset1.csv")
    print(f"  Shape: {dataset1.shape}")
except Exception as e:
    print(f" Error loading dataset1: {e}")
    dataset1 = None

try:
    dataset2 = pd.read_csv("dataset2.csv")
    print(" Successfully loaded dataset2.csv")
    print(f"  Shape: {dataset2.shape}")
except Exception as e:
    print(f" Error loading dataset2: {e}")
    dataset2 = None

# Display first few rows
print("\n--- Dataset 1 Preview ---")
print(dataset1.head())

print("\n--- Dataset 2 Preview ---")
print(dataset2.head())

# ============================================================================
# SECTION 2: DATA CLEANING AND PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: DATA CLEANING AND PREPROCESSING")
print("="*80)

# Check for missing values in dataset1
print("\n--- Missing Values in Dataset 1 ---")
missing_d1 = dataset1.isnull().sum()
print(missing_d1[missing_d1 > 0])
print(f"Total missing values: {dataset1.isnull().sum().sum()}")

# Check for missing values in dataset2
print("\n--- Missing Values in Dataset 2 ---")
missing_d2 = dataset2.isnull().sum()
print(missing_d2[missing_d2 > 0])
print(f"Total missing values: {dataset2.isnull().sum().sum()}")

# Handle missing values 
dataset1_clean = dataset1.dropna(subset=['bat_landing_to_food', 'seconds_after_rat_arrival'])
print(f"\n Dataset1: Removed {len(dataset1) - len(dataset1_clean)} rows with missing critical values")

# Check data types and ranges
print("\n--- Dataset 1 Summary Statistics ---")
print(dataset1_clean.describe())

print("\n--- Dataset 2 Summary Statistics ---")
print(dataset2.describe())

# ============================================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: FEATURE ENGINEERING")
print("="*80)

# Create new features for dataset1
dataset1_clean['vigilance_time'] = dataset1_clean['bat_landing_to_food']
dataset1_clean['season_label'] = dataset1_clean['season'].map({0: 'Winter', 1: 'Spring'})
dataset1_clean['risk_label'] = dataset1_clean['risk'].map({0: 'Risk-Avoidance', 1: 'Risk-Taking'})
dataset1_clean['reward_label'] = dataset1_clean['reward'].map({0: 'No Reward', 1: 'Reward'})

# Create time categories for dataset1
dataset1_clean['time_category'] = pd.cut(dataset1_clean['hours_after_sunset'], 
                                          bins=[0, 3, 6, 12], 
                                          labels=['Early Night', 'Mid Night', 'Late Night'])

# Create new features for dataset2
dataset2['season_label'] = dataset2['month'].apply(lambda x: 'Winter' if x <= 2 else 'Spring')
dataset2['rat_presence_intensity'] = pd.cut(dataset2['rat_minutes'], 
                                             bins=[0, 5, 15, 130], 
                                             labels=['Low', 'Medium', 'High'])
dataset2['food_depletion'] = 4 - dataset2['food_availability']  # Inverse of availability

print(" Created new features:")
print("  - vigilance_time (bat_landing_to_food)")
print("  - season_label (Winter/Spring)")
print("  - time_category (Early/Mid/Late Night)")
print("  - rat_presence_intensity (Low/Medium/High)")
print("  - food_depletion")

# ============================================================================
# SECTION 4: DATASET INTEGRATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: DATASET INTEGRATION")
print("="*80)

# Convert time columns to datetime for proper merging
dataset1_clean['start_time_dt'] = pd.to_datetime(dataset1_clean['start_time'], errors='coerce')
dataset2['time_dt'] = pd.to_datetime(dataset2['time'], errors='coerce')

# Create time windows for merging
dataset2['time_end'] = dataset2['time_dt'] + pd.Timedelta(minutes=30)

# Aggregate dataset2 by season for integration
dataset2_by_season = dataset2.groupby('season_label').agg({
    'bat_landing_number': ['mean', 'sum'],
    'rat_arrival_number': ['mean', 'sum'],
    'rat_minutes': ['mean', 'sum'],
    'food_availability': 'mean'
}).reset_index()

dataset2_by_season.columns = ['season_label', 'avg_bat_landings', 'total_bat_landings',
                               'avg_rat_arrivals', 'total_rat_arrivals',
                               'avg_rat_minutes', 'total_rat_minutes', 'avg_food_avail']

print("\n--- Aggregated Dataset2 by Season ---")
print(dataset2_by_season)

# Merge with dataset1
integrated_data = dataset1_clean.merge(dataset2_by_season, on='season_label', how='left')
print(f"\n Created integrated dataset with shape: {integrated_data.shape}")
print(f"  Combined individual bat behaviors with aggregated environmental data")