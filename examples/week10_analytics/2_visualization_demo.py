"""
Visualization Suite Demonstration

This script demonstrates the comprehensive visualization capabilities of SynFinance,
including static charts, interactive visualizations, geographic maps, and statistical plots.

Week 10 Day 2 - Comprehensive Visualization Suite

Author: SynFinance Team
Date: 2024
"""

# Configure matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualizations import (
    StaticCharts,
    InteractiveCharts,
    GeographicMaps,
    StatisticalPlots,
    VisualizationGallery,
    ExportManager,
    ChartTheme,
    ColorPalette,
)


def generate_sample_data(n_records=10000):
    """
    Generate sample financial transaction data for visualization demonstrations.
    
    Args:
        n_records: Number of transaction records to generate
        
    Returns:
        pd.DataFrame: Sample transaction data with realistic patterns
    """
    np.random.seed(42)
    
    # Generate dates over the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_records)
    
    # Generate transaction amounts with realistic distribution
    amounts = np.random.lognormal(mean=4, sigma=1.5, size=n_records)
    amounts = np.clip(amounts, 1, 10000)  # Clip extreme values
    
    # Generate merchant categories
    categories = np.random.choice(
        ['retail', 'dining', 'travel', 'entertainment', 'utilities', 'healthcare'],
        size=n_records,
        p=[0.3, 0.25, 0.15, 0.1, 0.15, 0.05]
    )
    
    # Generate geographic coordinates (US-focused)
    latitudes = np.random.normal(loc=39.8, scale=5, size=n_records)
    longitudes = np.random.normal(loc=-98.5, scale=15, size=n_records)
    
    # Generate customer segments
    segments = np.random.choice(
        ['premium', 'standard', 'basic'],
        size=n_records,
        p=[0.2, 0.5, 0.3]
    )
    
    # Generate fraud labels (realistic fraud rate)
    fraud_probability = np.where(
        amounts > 1000, 0.05,  # Higher fraud rate for large transactions
        np.where(categories == 'travel', 0.03, 0.01)  # Higher for travel
    )
    is_fraud = np.random.random(n_records) < fraud_probability
    
    # Create dataframe
    df = pd.DataFrame({
        'transaction_date': dates,
        'amount': amounts,
        'merchant_category': categories,
        'latitude': latitudes,
        'longitude': longitudes,
        'customer_segment': segments,
        'is_fraud': is_fraud,
        'transaction_count': np.random.poisson(lam=5, size=n_records),
        'customer_age': np.random.normal(loc=45, scale=15, size=n_records).astype(int),
        'account_age_days': np.random.uniform(30, 3650, n_records).astype(int),
    })
    
    # Add derived features
    df['day_of_week'] = df['transaction_date'].dt.dayofweek
    df['hour_of_day'] = np.random.randint(0, 24, n_records)
    df['log_amount'] = np.log1p(df['amount'])
    
    return df


def demo_static_charts(df, output_dir):
    """Demonstrate static chart capabilities."""
    print("\n" + "=" * 80)
    print("STATIC CHARTS DEMONSTRATION")
    print("=" * 80)
    
    charts = StaticCharts()
    export_mgr = ExportManager()
    
    # 1. Distribution analysis
    print("\n1. Creating distribution plots...")
    fig1 = charts.plot_distribution(
        df, 'amount',
        title='Transaction Amount Distribution',
        xlabel='Amount ($)',
        bins=50
    )
    export_mgr.save_figure(
        fig1,
        output_dir / 'static_distribution.png',
        formats=['png', 'svg']
    )
    print(f"   ✓ Saved distribution plot to {output_dir}")
    
    # 2. Boxplot for outlier detection
    print("\n2. Creating boxplot for outlier analysis...")
    fig2 = charts.plot_boxplot(
        df, y='amount',
        x='merchant_category',
        title='Transaction Amounts by Category',
        ylabel='Amount ($)'
    )
    export_mgr.save_figure(fig2, output_dir / 'static_boxplot.png')
    print(f"   ✓ Saved boxplot to {output_dir}")
    
    # 3. Correlation heatmap
    print("\n3. Creating correlation heatmap...")
    numeric_cols = ['amount', 'transaction_count', 'customer_age', 'account_age_days', 'day_of_week']
    fig3 = charts.plot_correlation_heatmap(
        df[numeric_cols],
        title='Feature Correlation Matrix',
        method='pearson'
    )
    export_mgr.save_figure(fig3, output_dir / 'static_correlation.png')
    print(f"   ✓ Saved correlation heatmap to {output_dir}")
    
    # 4. Time series analysis
    print("\n4. Creating time series plot...")
    daily_volume = df.groupby(df['transaction_date'].dt.date)['amount'].sum().reset_index()
    daily_volume.columns = ['date', 'total_amount']
    daily_volume['date'] = pd.to_datetime(daily_volume['date'])
    
    fig4 = charts.plot_time_series(
        daily_volume, 'date', 'total_amount',
        title='Daily Transaction Volume',
        xlabel='Date',
        ylabel='Total Amount ($)'
    )
    export_mgr.save_figure(fig4, output_dir / 'static_timeseries.png')
    print(f"   ✓ Saved time series plot to {output_dir}")
    
    # 5. Bar chart comparison
    print("\n5. Creating bar chart...")
    category_stats = df.groupby('merchant_category').agg({
        'amount': 'mean',
        'transaction_count': 'sum'
    }).reset_index()
    
    fig5 = charts.plot_bar_chart(
        category_stats, 'merchant_category', 'amount',
        title='Average Transaction Amount by Category',
        xlabel='Merchant Category',
        ylabel='Average Amount ($)'
    )
    export_mgr.save_figure(fig5, output_dir / 'static_barchart.png')
    print(f"   ✓ Saved bar chart to {output_dir}")
    
    # 6. Pie chart for composition
    print("\n6. Creating pie chart...")
    fraud_counts = df['is_fraud'].value_counts()
    fraud_counts.index = ['Legitimate', 'Fraudulent']
    fig6 = charts.plot_pie_chart(
        fraud_counts,
        title='Transaction Composition: Fraud vs Legitimate'
    )
    export_mgr.save_figure(fig6, output_dir / 'static_piechart.png')
    print(f"   ✓ Saved pie chart to {output_dir}")
    
    export_mgr.close_all()
    print("\n✓ Static charts demonstration complete!")


def demo_interactive_charts(df, output_dir):
    """Demonstrate interactive chart capabilities."""
    print("\n" + "=" * 80)
    print("INTERACTIVE CHARTS DEMONSTRATION")
    print("=" * 80)
    
    interactive = InteractiveCharts()
    
    # 1. Interactive histogram with statistics
    print("\n1. Creating interactive histogram...")
    fig1 = interactive.create_interactive_histogram(
        df, 'amount',
        title='Interactive Transaction Amount Distribution',
        nbins=50
    )
    interactive.save_html(fig1, output_dir / 'interactive_histogram.html')
    print(f"   ✓ Saved interactive histogram to {output_dir}")
    
    # 2. 3D scatter plot
    print("\n2. Creating 3D scatter plot...")
    fig2 = interactive.create_3d_scatter(
        df.sample(1000),  # Sample for performance
        x='amount',
        y='transaction_count',
        z='customer_age',
        color='merchant_category',
        title='3D Transaction Analysis'
    )
    interactive.save_html(fig2, output_dir / 'interactive_3d_scatter.html')
    print(f"   ✓ Saved 3D scatter plot to {output_dir}")
    
    # 3. Sunburst chart for hierarchical data
    print("\n3. Creating sunburst chart...")
    # Prepare hierarchical data
    hierarchy_data = df.groupby(['merchant_category', 'customer_segment']).size().reset_index(name='count')
    
    fig3 = interactive.create_sunburst(
        hierarchy_data,
        path=['merchant_category', 'customer_segment'],
        values='count',
        title='Transaction Hierarchy: Category → Segment'
    )
    interactive.save_html(fig3, output_dir / 'interactive_sunburst.html')
    print(f"   ✓ Saved sunburst chart to {output_dir}")
    
    # 4. Sankey diagram for flow analysis
    print("\n4. Creating Sankey diagram...")
    # Create flow data from category to segment
    flow_data = df.groupby(['merchant_category', 'customer_segment']).size().reset_index(name='value')
    
    # Map categories and segments to indices
    categories = df['merchant_category'].unique()
    segments = df['customer_segment'].unique()
    
    source = [list(categories).index(cat) for cat in flow_data['merchant_category']]
    target = [len(categories) + list(segments).index(seg) for seg in flow_data['customer_segment']]
    
    fig4 = interactive.create_sankey(
        source=source,
        target=target,
        value=flow_data['value'].tolist(),
        labels=list(categories) + list(segments),
        title='Transaction Flow: Category → Customer Segment'
    )
    interactive.save_html(fig4, output_dir / 'interactive_sankey.html')
    print(f"   ✓ Saved Sankey diagram to {output_dir}")
    
    # 5. Multi-line time series
    print("\n5. Creating multi-line time series...")
    daily_by_category = df.groupby([
        df['transaction_date'].dt.date,
        'merchant_category'
    ])['amount'].sum().reset_index()
    daily_by_category.columns = ['date', 'category', 'amount']
    daily_by_category['date'] = pd.to_datetime(daily_by_category['date'])
    
    # Pivot for multi-line format
    pivot_data = daily_by_category.pivot(index='date', columns='category', values='amount').reset_index()
    
    fig5 = interactive.create_multi_line(
        pivot_data, 'date',
        [col for col in pivot_data.columns if col != 'date'],
        title='Daily Transaction Volume by Category',
        xlabel='Date',
        ylabel='Amount ($)'
    )
    interactive.save_html(fig5, output_dir / 'interactive_multiline.html')
    print(f"   ✓ Saved multi-line chart to {output_dir}")
    
    print("\n✓ Interactive charts demonstration complete!")


def demo_geographic_maps(df, output_dir):
    """Demonstrate geographic map capabilities."""
    print("\n" + "=" * 80)
    print("GEOGRAPHIC MAPS DEMONSTRATION")
    print("=" * 80)
    
    geo_maps = GeographicMaps()
    
    # Sample data for performance
    sample_df = df.sample(min(1000, len(df)))
    
    # 1. Point map with markers
    print("\n1. Creating point map...")
    map1 = geo_maps.create_point_map(
        sample_df,
        'latitude',
        'longitude',
        popup_fields=['amount', 'merchant_category', 'is_fraud'],
        tooltip_fields=['amount', 'merchant_category']
    )
    geo_maps.save(map1, output_dir / 'geo_point_map.html')
    print(f"   ✓ Saved point map to {output_dir}")
    
    # 2. Heatmap for transaction density
    print("\n2. Creating heatmap...")
    map2 = geo_maps.create_heatmap(
        sample_df,
        'latitude',
        'longitude',
        intensity_col='amount',
        radius=15,
        blur=25
    )
    geo_maps.save(map2, output_dir / 'geo_heatmap.html')
    print(f"   ✓ Saved heatmap to {output_dir}")
    
    # 3. Cluster map for large datasets
    print("\n3. Creating cluster map...")
    map3 = geo_maps.create_cluster_map(
        sample_df,
        'latitude',
        'longitude',
        popup_fields=['amount', 'merchant_category']
    )
    geo_maps.save(map3, output_dir / 'geo_cluster_map.html')
    print(f"   ✓ Saved cluster map to {output_dir}")
    
    # 4. Fraud-specific visualization
    print("\n4. Creating fraud location map...")
    fraud_df = df[df['is_fraud'] == True].sample(min(200, df['is_fraud'].sum()))
    
    map4 = geo_maps.create_point_map(
        fraud_df,
        'latitude',
        'longitude',
        popup_fields=['amount', 'merchant_category', 'transaction_date'],
        marker_color='red'
    )
    geo_maps.save(map4, output_dir / 'geo_fraud_map.html')
    print(f"   ✓ Saved fraud location map to {output_dir}")
    
    print("\n✓ Geographic maps demonstration complete!")


def demo_statistical_plots(df, output_dir):
    """Demonstrate statistical plot capabilities."""
    print("\n" + "=" * 80)
    print("STATISTICAL PLOTS DEMONSTRATION")
    print("=" * 80)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    stat_plots = StatisticalPlots()
    export_mgr = ExportManager()
    
    # Prepare data for modeling
    feature_cols = ['amount', 'transaction_count', 'customer_age', 'account_age_days', 'day_of_week', 'hour_of_day']
    X = df[feature_cols].fillna(0)
    y = df['is_fraud'].astype(int)
    
    # Encode categorical features
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded['merchant_category_encoded'] = le.fit_transform(df['merchant_category'])
    
    X_with_cat = df_encoded[feature_cols + ['merchant_category_encoded']].fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_with_cat, y, test_size=0.3, random_state=42)
    
    # Train model
    print("\n1. Training fraud detection model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("   ✓ Model trained successfully")
    
    # 2. ROC curve
    print("\n2. Creating ROC curve...")
    fig1 = stat_plots.plot_roc_curve(
        y_test, y_proba,
        title='Fraud Detection Model - ROC Curve'
    )
    export_mgr.save_figure(fig1, output_dir / 'stat_roc_curve.png')
    print(f"   ✓ Saved ROC curve to {output_dir}")
    
    # 3. Confusion matrix
    print("\n3. Creating confusion matrix...")
    fig2 = stat_plots.plot_confusion_matrix(
        y_test, y_pred,
        labels=['Legitimate', 'Fraudulent'],
        title='Fraud Detection - Confusion Matrix',
        normalize=True
    )
    export_mgr.save_figure(fig2, output_dir / 'stat_confusion_matrix.png')
    print(f"   ✓ Saved confusion matrix to {output_dir}")
    
    # 4. Feature importance
    print("\n4. Creating feature importance plot...")
    feature_names = feature_cols + ['merchant_category']
    importance = np.abs(model.coef_[0])
    
    fig3 = stat_plots.plot_feature_importance(
        importance,
        feature_names,
        title='Fraud Detection - Feature Importance',
        top_n=10
    )
    export_mgr.save_figure(fig3, output_dir / 'stat_feature_importance.png')
    print(f"   ✓ Saved feature importance plot to {output_dir}")
    
    # 5. Learning curve
    print("\n5. Creating learning curve...")
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    fig4 = stat_plots.plot_learning_curve(
        train_sizes, train_scores, val_scores,
        title='Fraud Detection - Learning Curve',
        ylabel='ROC AUC'
    )
    export_mgr.save_figure(fig4, output_dir / 'stat_learning_curve.png')
    print(f"   ✓ Saved learning curve to {output_dir}")
    
    # 6. Regression analysis on amount prediction
    print("\n6. Creating regression plot...")
    from sklearn.linear_model import LinearRegression
    
    # Predict amount based on features
    X_reg = df[['transaction_count', 'customer_age', 'account_age_days']].fillna(0)
    y_reg = df['amount']
    
    reg_model = LinearRegression()
    reg_model.fit(X_reg, y_reg)
    y_reg_pred = reg_model.predict(X_reg)
    
    fig5 = stat_plots.plot_regression(
        y_reg[:1000], y_reg_pred[:1000],  # Sample for clarity
        title='Transaction Amount Prediction',
        xlabel='Actual Amount ($)',
        ylabel='Predicted Amount ($)'
    )
    export_mgr.save_figure(fig5, output_dir / 'stat_regression.png')
    print(f"   ✓ Saved regression plot to {output_dir}")
    
    # 7. Residuals analysis
    print("\n7. Creating residuals plot...")
    fig6 = stat_plots.plot_residuals(
        y_reg[:1000], y_reg_pred[:1000],
        title='Residuals Analysis - Amount Prediction'
    )
    export_mgr.save_figure(fig6, output_dir / 'stat_residuals.png')
    print(f"   ✓ Saved residuals plot to {output_dir}")
    
    export_mgr.close_all()
    print("\n✓ Statistical plots demonstration complete!")


def demo_gallery_dashboards(df, output_dir):
    """Demonstrate pre-configured dashboard galleries."""
    print("\n" + "=" * 80)
    print("GALLERY DASHBOARDS DEMONSTRATION")
    print("=" * 80)
    
    gallery = VisualizationGallery(df)
    
    # 1. Overview dashboard
    print("\n1. Generating overview dashboard...")
    figs1 = gallery.generate_overview_dashboard(
        output_dir=str(output_dir / 'dashboard_overview')
    )
    print(f"   ✓ Generated {len(figs1)} visualizations in overview dashboard")
    print(f"   ✓ Saved to {output_dir / 'dashboard_overview'}")
    
    # 2. Correlation dashboard
    print("\n2. Generating correlation dashboard...")
    figs2 = gallery.generate_correlation_dashboard(
        output_dir=str(output_dir / 'dashboard_correlation')
    )
    print(f"   ✓ Generated {len(figs2)} visualizations in correlation dashboard")
    print(f"   ✓ Saved to {output_dir / 'dashboard_correlation'}")
    
    # 3. Distribution dashboard
    print("\n3. Generating distribution dashboard...")
    figs3 = gallery.generate_distribution_dashboard(
        output_dir=str(output_dir / 'dashboard_distribution')
    )
    print(f"   ✓ Generated {len(figs3)} visualizations in distribution dashboard")
    print(f"   ✓ Saved to {output_dir / 'dashboard_distribution'}")
    
    # 4. Time series dashboard
    print("\n4. Generating time series dashboard...")
    figs4 = gallery.generate_time_series_dashboard(
        date_field='transaction_date',
        value_fields=['amount', 'transaction_count'],
        output_dir=str(output_dir / 'dashboard_timeseries')
    )
    print(f"   ✓ Generated {len(figs4)} visualizations in time series dashboard")
    print(f"   ✓ Saved to {output_dir / 'dashboard_timeseries'}")
    
    # 5. Fraud analysis dashboard
    print("\n5. Generating fraud analysis dashboard...")
    figs5 = gallery.generate_fraud_analysis_dashboard(
        fraud_field='is_fraud',
        amount_field='amount',
        category_field='merchant_category',
        output_dir=str(output_dir / 'dashboard_fraud')
    )
    print(f"   ✓ Generated {len(figs5)} visualizations in fraud dashboard")
    print(f"   ✓ Saved to {output_dir / 'dashboard_fraud'}")
    
    # 6. Complete gallery with HTML report
    print("\n6. Generating complete gallery with HTML report...")
    gallery_dir = output_dir / 'complete_gallery'
    gallery.save_gallery(
        str(gallery_dir),
        include_dashboards=[
            'overview',
            'correlation',
            'distribution',
            'time_series',
            'fraud_analysis'
        ]
    )
    
    html_path = gallery.create_html_report(
        str(gallery_dir),
        report_title='SynFinance Transaction Analysis Report'
    )
    print(f"   ✓ Complete gallery saved to {gallery_dir}")
    print(f"   ✓ HTML report generated: {html_path}")
    
    print("\n✓ Gallery dashboards demonstration complete!")


def demo_themes_and_styling():
    """Demonstrate theme and styling capabilities."""
    print("\n" + "=" * 80)
    print("THEMES AND STYLING DEMONSTRATION")
    print("=" * 80)
    
    from visualizations.themes import (
        get_theme,
        DEFAULT_PALETTE,
        COLORBLIND_PALETTE,
        DARK_PALETTE,
        SEQUENTIAL_PALETTES,
        DIVERGING_PALETTES
    )
    
    # 1. Available palettes
    print("\n1. Available color palettes:")
    palettes = {
        'default': DEFAULT_PALETTE,
        'colorblind': COLORBLIND_PALETTE,
        'dark': DARK_PALETTE,
    }
    for name, palette in palettes.items():
        print(f"   - {name}: {palette.description}")
        print(f"     Colors: {len(palette.colors)} colors")
    
    print("\n   Sequential palettes:")
    for name, palette in SEQUENTIAL_PALETTES.items():
        print(f"   - {name}: {palette.description}")
    
    print("\n   Diverging palettes:")
    for name, palette in DIVERGING_PALETTES.items():
        print(f"   - {name}: {palette.description}")
    
    # 2. Pre-defined themes
    print("\n2. Pre-defined themes:")
    theme_names = ['default', 'dark', 'colorblind', 'minimal', 'presentation']
    for theme_name in theme_names:
        theme = get_theme(theme_name)
        print(f"   - {theme_name}: {theme.palette.name} palette, {theme.style} style")
    
    # 3. Custom theme creation
    print("\n3. Creating custom theme...")
    custom_palette = ColorPalette(
        name='custom',
        colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
        description='Custom theme for demonstration'
    )
    custom_theme = ChartTheme(
        palette=custom_palette,
        style='whitegrid',
        context='notebook',
        font_scale=1.2
    )
    print(f"   ✓ Custom theme created with {len(custom_palette.colors)} colors")
    
    print("\n✓ Themes and styling demonstration complete!")


def main():
    """Main demonstration function."""
    print("\n" + "=" * 80)
    print("SYNFINANCE VISUALIZATION SUITE DEMONSTRATION")
    print("Week 10 Day 2 - Comprehensive Visualization Capabilities")
    print("=" * 80)
    
    # Setup output directory
    output_dir = Path(__file__).parent / 'output' / 'visualization_demo'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Generate sample data
    print("\nGenerating sample transaction data...")
    df = generate_sample_data(n_records=10000)
    print(f"✓ Generated {len(df):,} transaction records")
    print(f"  - Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"  - Amount range: ${df['amount'].min():.2f} to ${df['amount'].max():.2f}")
    print(f"  - Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"  - Categories: {', '.join(df['merchant_category'].unique())}")
    
    # Run demonstrations
    try:
        demo_themes_and_styling()
        demo_static_charts(df, output_dir)
        demo_interactive_charts(df, output_dir)
        demo_geographic_maps(df, output_dir)
        demo_statistical_plots(df, output_dir)
        demo_gallery_dashboards(df, output_dir)
        
        # Summary
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print(f"\nAll visualizations have been saved to: {output_dir}")
        print("\nGenerated files:")
        print("  Static Charts:")
        print("    - static_distribution.png/svg")
        print("    - static_boxplot.png")
        print("    - static_correlation.png")
        print("    - static_timeseries.png")
        print("    - static_barchart.png")
        print("    - static_piechart.png")
        print("\n  Interactive Charts:")
        print("    - interactive_histogram.html")
        print("    - interactive_3d_scatter.html")
        print("    - interactive_sunburst.html")
        print("    - interactive_sankey.html")
        print("    - interactive_multiline.html")
        print("\n  Geographic Maps:")
        print("    - geo_point_map.html")
        print("    - geo_heatmap.html")
        print("    - geo_cluster_map.html")
        print("    - geo_fraud_map.html")
        print("\n  Statistical Plots:")
        print("    - stat_roc_curve.png")
        print("    - stat_confusion_matrix.png")
        print("    - stat_feature_importance.png")
        print("    - stat_learning_curve.png")
        print("    - stat_regression.png")
        print("    - stat_residuals.png")
        print("\n  Dashboard Galleries:")
        print("    - dashboard_overview/")
        print("    - dashboard_correlation/")
        print("    - dashboard_distribution/")
        print("    - dashboard_timeseries/")
        print("    - dashboard_fraud/")
        print("    - complete_gallery/ (with HTML report)")
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
