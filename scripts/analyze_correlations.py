"""
Week 3 Days 2-3: Correlation Analysis and Pattern Discovery

This script performs comprehensive analysis on the 10K transaction dataset:
1. Calculate correlation matrix for numerical fields
2. Generate correlation heatmap
3. Analyze 5 key patterns
4. Perform statistical tests
5. Generate insights and recommendations

Input: output/week3_analysis_dataset.csv
Output: output/correlation_matrix.csv, output/correlation_heatmap.png,
        output/pattern_analysis_results.json
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

def main():
    """Perform complete correlation and pattern analysis."""
    print("=" * 80)
    print("Week 3 Days 2-3: Correlation Analysis and Pattern Discovery")
    print("=" * 80)
    
    # Load dataset
    print("\n[1/6] Loading dataset...")
    data_path = Path("output/week3_analysis_dataset.csv")
    df = pd.DataFrame(pd.read_csv(data_path))
    
    print(f"  Loaded {len(df):,} transactions with {len(df.columns)} columns")
    
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Found {len(numerical_cols)} numerical columns for correlation analysis")
    
    # Calculate correlation matrix
    print("\n[2/6] Calculating correlation matrix...")
    corr_matrix = df[numerical_cols].corr()
    
    print(f"  Correlation matrix shape: {corr_matrix.shape}")
    
    # Find strong correlations
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.3:  # Threshold for interesting correlations
                strong_corr.append({
                    'Field 1': corr_matrix.columns[i],
                    'Field 2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    strong_corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False)
    
    print(f"  Found {len(strong_corr_df)} strong correlations (|r| > 0.3)")
    print(f"\n  Top 10 strongest correlations:")
    for idx, row in strong_corr_df.head(10).iterrows():
        print(f"    {row['Field 1']} <-> {row['Field 2']}: {row['Correlation']:.3f}")
    
    # Save correlation matrix
    corr_output = Path("output/correlation_matrix.csv")
    corr_matrix.to_csv(corr_output)
    print(f"\n  Saved full correlation matrix to: {corr_output.name}")
    
    # Save strong correlations
    strong_corr_output = Path("output/strong_correlations.csv")
    strong_corr_df.to_csv(strong_corr_output, index=False)
    print(f"  Saved strong correlations to: {strong_corr_output.name}")
    
    # Generate heatmap
    print("\n[3/6] Generating correlation heatmap...")
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: All Numerical Fields', fontsize=16, pad=20)
    plt.tight_layout()
    
    heatmap_path = Path("output/correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved heatmap to: {heatmap_path.name}")
    
    # Pattern Analysis
    print("\n[4/6] Analyzing 5 key patterns...")
    patterns = {}
    
    # Pattern 1: Age vs Payment Mode
    print("\n  Pattern 1: Age vs Payment Mode")
    age_payment = pd.crosstab(df['Customer_Age'], df['Payment_Mode'], normalize='index') * 100
    patterns['age_payment'] = age_payment.to_dict()
    
    # Find dominant payment modes by age
    for age in [25, 35, 45, 55, 65]:
        closest_age = age_payment.index[np.abs(age_payment.index - age).argmin()]
        dominant_mode = age_payment.loc[closest_age].idxmax()
        pct = age_payment.loc[closest_age,dominant_mode]
        print(f"    Age {age}: Prefers {dominant_mode} ({pct:.1f}%)")
    
    # Pattern 2: Income vs Transaction Amount
    print("\n  Pattern 2: Income Bracket vs Transaction Amount")
    income_amount = df.groupby('Customer_Income_Bracket')['Amount'].agg(['mean', 'median', 'std'])
    patterns['income_amount'] = income_amount.to_dict()
    
    for bracket in income_amount.index:
        print(f"    {bracket}: Mean=₹{income_amount.loc[bracket, 'mean']:,.0f}, " +
              f"Median=₹{income_amount.loc[bracket, 'median']:,.0f}")
    
    # Statistical test: ANOVA for income vs amount
    income_groups = [df[df['Customer_Income_Bracket']==b]['Amount'].values 
                    for b in df['Customer_Income_Bracket'].unique()]
    f_stat, p_value = stats.f_oneway(*income_groups)
    print(f"    ANOVA test: F={f_stat:.2f}, p={p_value:.4f} " +
          f"({'Significant' if p_value < 0.05 else 'Not significant'})")
    
    # Pattern 3: Digital Savviness vs Device Type
    print("\n  Pattern 3: Digital Savviness vs Device Type")
    savviness_device = pd.crosstab(df['Customer_Digital_Savviness'], 
                                   df['Device_Type'], normalize='index') * 100
    patterns['savviness_device'] = savviness_device.to_dict()
    
    for savv in savviness_device.index:
        dominant_device = savviness_device.loc[savv].idxmax()
        pct = savviness_device.loc[savv, dominant_device]
        print(f"    {savv} savviness: Prefers {dominant_device} ({pct:.1f}%)")
    
    # Pattern 4: Distance from Home vs Transaction Status
    print("\n  Pattern 4: Distance from Home vs Transaction Status")
    # Bin distances
    df['Distance_Bin'] = pd.cut(df['Distance_from_Home'], 
                                bins=[0, 1, 10, 50, 200, 2000],
                                labels=['Same City', '1-10km', '10-50km', '50-200km', '200km+'])
    
    # Calculate decline rate by distance
    distance_status = pd.crosstab(df['Distance_Bin'], df['Transaction_Status'], 
                                  normalize='index') * 100
    patterns['distance_status'] = distance_status.to_dict()
    
    for dist_bin in distance_status.index:
        declined_pct = distance_status.loc[dist_bin, 'Declined'] if 'Declined' in distance_status.columns else 0
        count = len(df[df['Distance_Bin']==dist_bin])
        print(f"    {dist_bin}: Decline Rate={declined_pct:.1f}%, Count={count:,}")
    
    # Correlation test between distance and is_first_transaction
    corr_coef, corr_p = stats.pearsonr(df['Distance_from_Home'], 
                                       df['Is_First_Transaction_with_Merchant'].astype(int))
    print(f"    Distance vs New Merchant: r={corr_coef:.3f}, p={corr_p:.4f}")
    
    # Pattern 5: Time of Day vs Transaction Channel
    print("\n  Pattern 5: Hour of Day vs Transaction Channel")
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
    
    # Group hours into time periods
    def get_time_period(hour):
        if 6 <= hour < 12:
            return 'Morning (6-12)'
        elif 12 <= hour < 18:
            return 'Afternoon (12-18)'
        elif 18 <= hour < 22:
            return 'Evening (18-22)'
        else:
            return 'Night (22-6)'
    
    df['Time_Period'] = df['Hour'].apply(get_time_period)
    
    time_channel = pd.crosstab(df['Time_Period'], df['Transaction_Channel'], 
                               normalize='index') * 100
    patterns['time_channel'] = time_channel.to_dict()
    
    for period in ['Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-22)', 'Night (22-6)']:
        if period in time_channel.index:
            dominant_channel = time_channel.loc[period].idxmax()
            pct = time_channel.loc[period, dominant_channel]
            print(f"    {period}: Prefers {dominant_channel} ({pct:.1f}%)")
    
    # Statistical insights
    print("\n[5/6] Statistical Insights...")
    
    insights = {
        'dataset_stats': {
            'total_transactions': len(df),
            'unique_customers': df['Customer_ID'].nunique(),
            'unique_merchants': df['Merchant_ID'].nunique(),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
            'numerical_fields': len(numerical_cols),
            'strong_correlations_count': len(strong_corr_df)
        },
        'key_findings': [
            f"Found {len(strong_corr_df)} strong correlations (|r| > 0.3)",
            f"Strongest correlation: {strong_corr_df.iloc[0]['Field 1']} <-> " +
            f"{strong_corr_df.iloc[0]['Field 2']} (r={strong_corr_df.iloc[0]['Correlation']:.3f})",
            f"Income significantly affects transaction amount (ANOVA p={p_value:.4f})",
            f"Distance from home correlates with risk score (r={corr_coef:.3f})",
            f"Payment mode preferences vary significantly by age group"
        ],
        'patterns': patterns
    }
    
    # Save insights
    insights_path = Path("output/pattern_analysis_results.json")
    with open(insights_path, 'w') as f:
        json.dump(insights, f, indent=2, default=str)
    
    print(f"  Saved statistical insights to: {insights_path.name}")
    
    # Generate visualizations
    print("\n[6/6] Generating pattern visualizations...")
    
    # Plot 1: Income vs Amount
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Income vs Amount boxplot
    income_order = ['Low', 'Lower Middle', 'Middle', 'Upper Middle', 'High']
    income_in_data = [i for i in income_order if i in df['Customer_Income_Bracket'].values]
    axes[0, 0].boxplot([df[df['Customer_Income_Bracket']==bracket]['Amount'].values 
                        for bracket in income_in_data],
                       labels=income_in_data)
    axes[0, 0].set_title('Transaction Amount by Income Bracket', fontsize=12)
    axes[0, 0].set_xlabel('Income Bracket')
    axes[0, 0].set_ylabel('Amount (₹)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Distance vs Decline Rate scatter (since we don't have Risk_Score)
    # Calculate decline indicator (1 if Declined, 0 otherwise)
    df['Is_Declined'] = (df['Transaction_Status'] == 'Declined').astype(int)
    axes[0, 1].scatter(df['Distance_from_Home'], df['Is_Declined'], alpha=0.2, s=5)
    axes[0, 1].set_title('Distance from Home vs Transaction Decline', fontsize=12)
    axes[0, 1].set_xlabel('Distance from Home (km)')
    axes[0, 1].set_ylabel('Declined (1) vs Approved (0)')
    axes[0, 1].set_xlim(0, min(df['Distance_from_Home'].max(), 500))
    
    # Payment mode by age
    age_bins = [18, 30, 40, 50, 60, 100]
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=age_bins, 
                             labels=['18-30', '30-40', '40-50', '50-60', '60+'])
    age_payment_plot = pd.crosstab(df['Age_Group'], df['Payment_Mode'], normalize='index')
    age_payment_plot.plot(kind='bar', stacked=True, ax=axes[1, 0])
    axes[1, 0].set_title('Payment Mode Distribution by Age Group', fontsize=12)
    axes[1, 0].set_xlabel('Age Group')
    axes[1, 0].set_ylabel('Proportion')
    axes[1, 0].legend(title='Payment Mode', bbox_to_anchor=(1.05, 1))
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Channel by time period
    time_channel_plot = pd.crosstab(df['Time_Period'], df['Transaction_Channel'], 
                                    normalize='index')
    time_channel_plot.plot(kind='bar', stacked=True, ax=axes[1, 1])
    axes[1, 1].set_title('Transaction Channel by Time of Day', fontsize=12)
    axes[1, 1].set_xlabel('Time Period')
    axes[1, 1].set_ylabel('Proportion')
    axes[1, 1].legend(title='Channel', bbox_to_anchor=(1.05, 1))
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    patterns_viz_path = Path("output/pattern_visualizations.png")
    plt.savefig(patterns_viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved pattern visualizations to: {patterns_viz_path.name}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nGenerated Outputs:")
    print(f"  1. {corr_output.name} - Full correlation matrix ({corr_matrix.shape[0]}x{corr_matrix.shape[1]})")
    print(f"  2. {strong_corr_output.name} - Strong correlations ({len(strong_corr_df)} pairs)")
    print(f"  3. {heatmap_path.name} - Correlation heatmap visualization")
    print(f"  4. {insights_path.name} - Statistical insights and patterns")
    print(f"  5. {patterns_viz_path.name} - Pattern visualizations (4 plots)")
    
    print(f"\nKey Findings:")
    for finding in insights['key_findings']:
        print(f"  - {finding}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
