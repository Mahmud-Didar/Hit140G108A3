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

# ============================================================================
# SECTION 5: INVESTIGATION A - PREDATOR PERCEPTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: INVESTIGATION A - DO BATS PERCEIVE RATS AS PREDATORS?")
print("="*80)

print("\nHYPOTHESES:")
print("-" * 80)
print("H1a (Vigilance): Bats show increased vigilance (longer bat_landing_to_food)")
print("                 when rats are present, indicating predator perception.")
print("H1b (Avoidance): Bats are more likely to exhibit risk-avoidance behavior")
print("                 in the presence of rats.")
print("H1c (Timing):    Bats land later after rat arrival (higher seconds_after_rat_arrival)")
print("                 when perceiving rats as threats.")
print("-" * 80)

# === ANALYSIS 5.1: Vigilance Behavior ===
print("\n--- Analysis 5.1: Vigilance Behavior (bat_landing_to_food) ---")

risk_takers = dataset1_clean[dataset1_clean['risk'] == 1]['bat_landing_to_food'].dropna()
risk_avoiders = dataset1_clean[dataset1_clean['risk'] == 0]['bat_landing_to_food'].dropna()

print(f"\nDescriptive Statistics:")
print(f"Risk-Takers:   n={len(risk_takers)}, Mean={risk_takers.mean():.2f}s, SD={risk_takers.std():.2f}s")
print(f"Risk-Avoiders: n={len(risk_avoiders)}, Mean={risk_avoiders.mean():.2f}s, SD={risk_avoiders.std():.2f}s")

# T-test
t_stat, p_value = ttest_ind(risk_avoiders, risk_takers, equal_var=False)
print(f"\nIndependent T-Test:")
print(f"  t-statistic = {t_stat:.3f}")
print(f"  p-value = {p_value:.4f}")
if p_value < 0.05:
    print(f"  Result: Significant difference (p < 0.05)")
else:
    print(f"  Result: No significant difference (p >= 0.05)")

# Mann-Whitney U test (non-parametric)
u_stat, p_value_mw = mannwhitneyu(risk_avoiders, risk_takers, alternative='two-sided')
print(f"\nMann-Whitney U Test (non-parametric):")
print(f"  U-statistic = {u_stat:.3f}")
print(f"  p-value = {p_value_mw:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((risk_takers.std()**2 + risk_avoiders.std()**2) / 2)
cohens_d = (risk_avoiders.mean() - risk_takers.mean()) / pooled_std
print(f"\nEffect Size (Cohen's d) = {cohens_d:.3f}")
if abs(cohens_d) < 0.2:
    print("  Interpretation: Small effect")
elif abs(cohens_d) < 0.5:
    print("  Interpretation: Medium effect")
else:
    print("  Interpretation: Large effect")

# === ANALYSIS 5.2: Risk-Taking Behavior ===
print("\n--- Analysis 5.2: Risk-Taking Behavior Patterns ---")

# Crosstab for risk vs reward
risk_reward_ct = pd.crosstab(dataset1_clean['risk'], dataset1_clean['reward'], 
                              margins=True, margins_name='Total')
print("\nCrosstab: Risk vs Reward")
print(risk_reward_ct)

# Chi-square test
chi2, p_chi, dof, expected = chi2_contingency(risk_reward_ct.iloc[:-1, :-1])
print(f"\nChi-Square Test of Independence:")
print(f" Chi-Square Test:  = {chi2:.3f}")
print(f"  p-value = {p_chi:.4f}")
print(f"  degrees of freedom = {dof}")
if p_chi < 0.05:
    print(f"   Result: Significant association (p < 0.05)")
else:
    print(f"   Result: No significant association (p >= 0.05)")

# Calculate proportions
prop_table = pd.crosstab(dataset1_clean['risk'], dataset1_clean['reward'], 
                         normalize='index') * 100
print("\nProportions (%):")
print(prop_table.round(2))

# === ANALYSIS 5.3: Timing of Bat Landing ===
print("\n--- Analysis 5.3: Timing After Rat Arrival ---")

print("\nDescriptive Statistics: seconds_after_rat_arrival")
print(dataset1_clean.groupby('risk')['seconds_after_rat_arrival'].describe())

# T-test for timing
timing_risk_takers = dataset1_clean[dataset1_clean['risk'] == 1]['seconds_after_rat_arrival'].dropna()
timing_risk_avoiders = dataset1_clean[dataset1_clean['risk'] == 0]['seconds_after_rat_arrival'].dropna()

t_stat_timing, p_value_timing = ttest_ind(timing_risk_avoiders, timing_risk_takers, equal_var=False)
print(f"\nIndependent T-Test:")
print(f"  t-statistic = {t_stat_timing:.3f}")
print(f"  p-value = {p_value_timing:.4f}")

# === ANALYSIS 5.4: Logistic Regression ===
print("\n--- Analysis 5.4: Logistic Regression (Predicting Risk Behavior) ---")

# Prepare data for logistic regression
log_data = dataset1_clean[['risk', 'seconds_after_rat_arrival', 
                            'hours_after_sunset', 'season']].dropna()

X = log_data[['seconds_after_rat_arrival', 'hours_after_sunset', 'season']]
X = sm.add_constant(X)
y = log_data['risk']

# Fit logistic regression model
logit_model = sm.Logit(y, X).fit(disp=0)
print(logit_model.summary())

print("\nInterpretation:")
print("  Positive coefficients increase odds of risk-taking behavior")
print("  Negative coefficients decrease odds of risk-taking behavior")



# === ANALYSIS 5.5: OLS Regression - Predicting Vigilance Time ===

print("\n" + "="*80)
# print("="*80)

integrated_data['hours_squared'] = integrated_data['hours_after_sunset'] ** 2
integrated_data['log_rat_minutes'] = np.log1p(integrated_data['avg_rat_minutes'])  # log(1 + x) to avoid log(0)

integrated_data['season_label'] = integrated_data['season_label'].astype('category')
integrated_data['time_category'] = integrated_data['time_category'].astype('category')

z_scores = np.abs(zscore(integrated_data['bat_landing_to_food']))
filtered_data = integrated_data[z_scores < 3]  # remove extreme outliers


ols_formula = (
    'bat_landing_to_food ~ '
    'seconds_after_rat_arrival + hours_after_sunset + np.power(hours_after_sunset, 2) + '
    'np.log1p(avg_rat_minutes) + risk + reward + '
    'C(season_label) + C(time_category) + '
    'reward:hours_after_sunset + seconds_after_rat_arrival:risk'
)


ols_model = ols(ols_formula, data=filtered_data.dropna()).fit()

print(ols_model.summary())

print("\nModel R² = {:.3f}".format(ols_model.rsquared))
print("Adjusted R² = {:.3f}".format(ols_model.rsquared_adj))


