"""
Demo: Automated Reporting System
==================================

Demonstrates the comprehensive reporting capabilities of SynFinance:
1. HTML Report Generation (Executive, Technical, Fraud, Quality)
2. Excel Dashboard Creation (Multi-sheet workbooks)
3. PDF Export (optional - requires WeasyPrint with GTK)
4. Chart Integration with Reports

This script generates professional reports in multiple formats from synthetic
transaction data, showcasing the reporting module's capabilities.

Author: SynFinance Development Team
Date: November 2, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import seaborn as sns

# Import reporting modules
from src.reporting import (
    HTMLReportGenerator,
    ExcelDashboardGenerator,
    PDF_AVAILABLE
)


def generate_sample_data(n_rows=1000):
    """Generate sample transaction data for demonstration."""
    np.random.seed(42)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_rows)
    
    # Generate transaction amounts
    amounts = np.random.lognormal(mean=8, sigma=1.5, size=n_rows)
    
    # Generate categories
    categories = np.random.choice(
        ['Electronics', 'Groceries', 'Travel', 'Entertainment', 'Healthcare'],
        size=n_rows,
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )
    
    # Generate fraud data (5% fraud rate)
    fraud_mask = np.random.random(n_rows) < 0.05
    fraud_types = np.where(
        fraud_mask,
        np.random.choice(['Card Skimming', 'Identity Theft', 'Account Takeover'], size=n_rows),
        None
    )
    
    # Generate customer and merchant IDs
    customer_ids = np.random.randint(1000, 2000, size=n_rows)
    merchant_ids = np.random.randint(500, 700, size=n_rows)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Transaction_ID': range(1, n_rows + 1),
        'Transaction_Date': dates,
        'Transaction_Amount': amounts,
        'Category': categories,
        'Customer_ID': customer_ids,
        'Merchant_ID': merchant_ids,
        'Fraud_Type': fraud_types,
        'Is_Fraud': fraud_mask.astype(int)
    })
    
    return data


def create_sample_charts(data):
    """Create sample charts for reports."""
    charts = {}
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Distribution chart (Transaction Amount)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    data['Transaction_Amount'].hist(bins=50, ax=ax1, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Transaction Amount ()', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    charts['distribution'] = fig1
    
    # 2. Category breakdown (Bar chart)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    category_counts = data['Category'].value_counts()
    category_counts.plot(kind='bar', ax=ax2, color='teal', edgecolor='black')
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_ylabel('Transaction Count', fontsize=12)
    ax2.set_title('Transactions by Category', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    charts['category_breakdown'] = fig2
    
    # 3. Fraud patterns (Stacked bar chart)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fraud_data = data[data['Is_Fraud'] == 1]
    if len(fraud_data) > 0:
        fraud_by_type = fraud_data['Fraud_Type'].value_counts()
        fraud_by_type.plot(kind='bar', ax=ax3, color='crimson', edgecolor='black')
        ax3.set_xlabel('Fraud Type', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('Fraud Patterns', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
    charts['fraud_patterns'] = fig3
    
    return charts


def calculate_metrics(data):
    """Calculate comprehensive metrics for reports."""
    metrics = {
        # Basic metrics
        'total_transactions': len(data),
        'total_amount': data['Transaction_Amount'].sum(),
        'avg_amount': data['Transaction_Amount'].mean(),
        'median_amount': data['Transaction_Amount'].median(),
        'min_amount': data['Transaction_Amount'].min(),
        'max_amount': data['Transaction_Amount'].max(),
        
        # Fraud metrics
        'fraud_count': data['Is_Fraud'].sum(),
        'fraud_rate': data['Is_Fraud'].sum() / len(data) if len(data) > 0 else 0,
        'fraud_amount': data[data['Is_Fraud'] == 1]['Transaction_Amount'].sum(),
        
        # Customer metrics
        'unique_customers': data['Customer_ID'].nunique(),
        'unique_merchants': data['Merchant_ID'].nunique(),
        'unique_categories': data['Category'].nunique(),
        
        # Data quality metrics
        'completeness': 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
        'variance': data['Transaction_Amount'].var(),
        'date_range_days': (data['Transaction_Date'].max() - data['Transaction_Date'].min()).days
    }
    
    return metrics


def generate_findings(data, metrics):
    """Generate key findings for reports."""
    findings = []
    
    # High fraud rate finding
    if metrics['fraud_rate'] > 0.03:
        findings.append({
            'title': 'Elevated Fraud Rate',
            'description': f"Fraud rate of {metrics['fraud_rate']:.1%} exceeds normal threshold of 3%",
            'severity': 'warning',
            'impact': 'Potential revenue loss and customer trust issues'
        })
    
    # High-value transactions
    high_value_count = (data['Transaction_Amount'] > 10000).sum()
    if high_value_count > len(data) * 0.1:
        findings.append({
            'title': 'High-Value Transactions',
            'description': f"{high_value_count} transactions exceed 10,000",
            'severity': 'info',
            'impact': 'Review required for large transactions'
        })
    
    # Category concentration
    top_category_pct = data['Category'].value_counts().iloc[0] / len(data)
    if top_category_pct > 0.4:
        findings.append({
            'title': 'Category Concentration',
            'description': f"Top category represents {top_category_pct:.1%} of all transactions",
            'severity': 'info',
            'impact': 'Business may be dependent on single category'
        })
    
    return findings


def generate_recommendations(data, metrics):
    """Generate recommendations for reports."""
    recommendations = []
    
    if metrics['fraud_rate'] > 0.03:
        recommendations.append(
            "Implement enhanced fraud detection rules for high-risk categories"
        )
        recommendations.append(
            "Review and update customer authentication mechanisms"
        )
    
    if metrics['avg_amount'] > 5000:
        recommendations.append(
            "Consider tiered transaction limits based on customer history"
        )
    
    if metrics['unique_customers'] < len(data) / 10:
        recommendations.append(
            "High transaction frequency per customer detected - monitor for unusual patterns"
        )
    
    recommendations.append(
        "Continue regular data quality monitoring and validation"
    )
    
    return recommendations


def demo_html_reports(data, metrics, charts, output_dir):
    """Demonstrate HTML report generation."""
    print("\n" + "="*70)
    print("HTML REPORT GENERATION")
    print("="*70)
    
    generator = HTMLReportGenerator()
    findings = generate_findings(data, metrics)
    recommendations = generate_recommendations(data, metrics)
    
    # 1. Executive Summary Report
    print("\n1. Generating Executive Summary Report...")
    html_exec = generator.generate_executive_report(
        data=data,
        metrics=metrics,
        charts=charts,
        findings=findings,
        recommendations=recommendations,
        output_path=output_dir / 'executive_summary.html'
    )
    print(f"    Generated: executive_summary.html ({len(html_exec):,} characters)")
    
    # 2. Technical Analysis Report
    print("\n2. Generating Technical Analysis Report...")
    statistics = {
        'mean': data['Transaction_Amount'].mean(),
        'median': data['Transaction_Amount'].median(),
        'std': data['Transaction_Amount'].std(),
        'variance': data['Transaction_Amount'].var(),
        'skewness': data['Transaction_Amount'].skew(),
        'kurtosis': data['Transaction_Amount'].kurtosis()
    }
    
    correlations = data[['Transaction_Amount', 'Is_Fraud']].corr()
    
    html_tech = generator.generate_technical_report(
        data=data,
        statistics=statistics,
        correlations=correlations,
        output_path=output_dir / 'technical_analysis.html'
    )
    print(f"    Generated: technical_analysis.html ({len(html_tech):,} characters)")
    
    # 3. Fraud Detection Report
    print("\n3. Generating Fraud Detection Report...")
    fraud_data = data[data['Is_Fraud'] == 1]
    fraud_statistics = {
        'total_fraud': len(fraud_data),
        'fraud_rate': metrics['fraud_rate'],
        'fraud_amount': fraud_data['Transaction_Amount'].sum(),
        'avg_fraud_amount': fraud_data['Transaction_Amount'].mean() if len(fraud_data) > 0 else 0,
        'patterns_detected': fraud_data['Fraud_Type'].nunique() if len(fraud_data) > 0 else 0
    }
    
    pattern_analysis = fraud_data['Fraud_Type'].value_counts().to_dict() if len(fraud_data) > 0 else {}
    
    html_fraud = generator.generate_fraud_report(
        data=data,
        fraud_statistics=fraud_statistics,
        pattern_analysis=pattern_analysis,
        output_path=output_dir / 'fraud_detection.html'
    )
    print(f"    Generated: fraud_detection.html ({len(html_fraud):,} characters)")
    
    # 4. Data Quality Report
    print("\n4. Generating Data Quality Report...")
    completeness = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
    quality_metrics = {
        'completeness': completeness,
        'variance': data['Transaction_Amount'].var()
    }
    field_analysis = {
        'Transaction_Amount': {
            'valid_range': (data['Transaction_Amount'] > 0).sum() / len(data),
            'outliers': ((data['Transaction_Amount'] - data['Transaction_Amount'].mean()).abs() > 3 * data['Transaction_Amount'].std()).sum()
        }
    }
    
    html_quality = generator.generate_quality_report(
        data=data,
        quality_metrics=quality_metrics,
        field_analysis=field_analysis,
        output_path=output_dir / 'data_quality.html'
    )
    print(f"    Generated: data_quality.html ({len(html_quality):,} characters)")
    
    print(f"\n All HTML reports saved to: {output_dir}/")


def demo_excel_dashboards(data, output_dir):
    """Demonstrate Excel dashboard generation."""
    print("\n" + "="*70)
    print("EXCEL DASHBOARD GENERATION")
    print("="*70)
    
    generator = ExcelDashboardGenerator()
    
    # 1. Full dashboard with all features
    print("\n1. Creating Full Dashboard (5 sheets)...")
    output_path = output_dir / 'transaction_dashboard.xlsx'
    generator.create_dashboard_workbook(
        data=data,
        output_path=output_path,
        include_fraud=True,
        include_charts=True
    )
    file_size = output_path.stat().st_size / 1024
    print(f"    Generated: transaction_dashboard.xlsx ({file_size:.1f} KB)")
    print(f"    Sheets: Summary Dashboard, Transaction Data, Statistical Analysis,")
    print(f"              Fraud Analysis, Charts & Visualizations")
    
    # 2. Dashboard without fraud analysis
    print("\n2. Creating Dashboard without Fraud Analysis...")
    output_path_no_fraud = output_dir / 'transaction_dashboard_no_fraud.xlsx'
    clean_data = data.copy()
    clean_data['Fraud_Type'] = None  # Remove fraud data
    clean_data['Is_Fraud'] = 0
    
    generator.create_dashboard_workbook(
        data=clean_data,
        output_path=output_path_no_fraud,
        include_fraud=False,
        include_charts=True
    )
    file_size_no_fraud = output_path_no_fraud.stat().st_size / 1024
    print(f"    Generated: transaction_dashboard_no_fraud.xlsx ({file_size_no_fraud:.1f} KB)")
    
    print(f"\n All Excel dashboards saved to: {output_dir}/")


def demo_pdf_export(output_dir):
    """Demonstrate PDF export (if available)."""
    print("\n" + "="*70)
    print("PDF EXPORT")
    print("="*70)
    
    if PDF_AVAILABLE:
        try:
            from src.reporting import PDFExporter
            
            print("\n PDF export is available")
            print("   Converting HTML reports to PDF...")
            
            exporter = PDFExporter()
            
            # Convert executive summary to PDF
            html_path = output_dir / 'executive_summary.html'
            pdf_path = output_dir / 'executive_summary.pdf'
            
            exporter.html_file_to_pdf(
                html_path=html_path,
                output_path=pdf_path
            )
            
            file_size = pdf_path.stat().st_size / 1024
            print(f"    Generated: executive_summary.pdf ({file_size:.1f} KB)")
            
        except Exception as e:
            print(f"    PDF export failed: {e}")
            print("   Note: WeasyPrint requires GTK libraries on Windows")
            print("   See: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer")
    else:
        print("\n PDF export not available")
        print("   Install WeasyPrint: pip install weasyprint")
        print("   Note: Requires GTK libraries on Windows")


def print_summary(data, metrics, output_dir):
    """Print demonstration summary."""
    print("\n" + "="*70)
    print("DEMONSTRATION SUMMARY")
    print("="*70)
    
    print(f"\n Dataset Statistics:")
    print(f"    Total Transactions: {metrics['total_transactions']:,}")
    print(f"    Total Amount: {metrics['total_amount']:,.2f}")
    print(f"    Average Amount: {metrics['avg_amount']:,.2f}")
    print(f"    Fraud Rate: {metrics['fraud_rate']:.2%}")
    print(f"    Unique Customers: {metrics['unique_customers']:,}")
    print(f"    Date Range: {metrics['date_range_days']} days")
    
    print(f"\n Generated Files:")
    for file in sorted(output_dir.glob('*')):
        file_size = file.stat().st_size / 1024
        print(f"    {file.name:<40} ({file_size:>8.1f} KB)")
    
    print(f"\n Demonstration Complete!")
    print(f"   All reports saved to: {output_dir}/")
    print(f"\n Next Steps:")
    print(f"   1. Open HTML reports in a web browser")
    print(f"   2. Open Excel dashboards in Microsoft Excel or LibreOffice")
    print(f"   3. Review generated charts and metrics")
    print(f"   4. Customize templates in src/reporting/templates/")


def main():
    """Main demonstration function."""
    print("="*70)
    print("SynFinance Automated Reporting System - Demonstration")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Version: 2.17.0")
    
    # Create output directory
    output_dir = Path('output/reports_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n Output Directory: {output_dir}/")
    
    # Generate sample data
    print("\n Generating sample transaction data...")
    data = generate_sample_data(n_rows=1000)
    print(f"    Generated {len(data):,} transactions")
    
    # Calculate metrics
    print("\n Calculating metrics...")
    metrics = calculate_metrics(data)
    print(f"    Calculated {len(metrics)} metrics")
    
    # Create charts
    print("\n Creating visualization charts...")
    charts = create_sample_charts(data)
    print(f"    Created {len(charts)} charts")
    
    # Demonstrate HTML reports
    demo_html_reports(data, metrics, charts, output_dir)
    
    # Demonstrate Excel dashboards
    demo_excel_dashboards(data, output_dir)
    
    # Demonstrate PDF export (if available)
    demo_pdf_export(output_dir)
    
    # Print summary
    print_summary(data, metrics, output_dir)
    
    # Close all matplotlib figures
    plt.close('all')


if __name__ == '__main__':
    main()
