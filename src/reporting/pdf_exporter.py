"""
PDF Export Module

This module provides HTML to PDF conversion capabilities using WeasyPrint.
Supports professional PDF generation with headers, footers, and page numbering.

Author: SynFinance Development Team
Date: November 2, 2025
Version: 2.17.0
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration


class PDFExporter:
    """
    Exports HTML reports to PDF format with professional styling.
    
    Features:
    - HTML to PDF conversion
    - Custom headers and footers
    - Page numbering
    - Print optimization
    - Professional styling
    """
    
    def __init__(self):
        """Initialize the PDF exporter."""
        self.font_config = FontConfiguration()
        
        # Default CSS for PDF print optimization
        self.default_print_css = """
            @page {
                size: A4;
                margin: 25mm;
                
                @top-center {
                    content: "SynFinance Report";
                    font-family: Arial, sans-serif;
                    font-size: 10pt;
                    color: #666;
                }
                
                @bottom-center {
                    content: "Page " counter(page) " of " counter(pages);
                    font-family: Arial, sans-serif;
                    font-size: 9pt;
                    color: #666;
                }
            }
            
            body {
                font-family: Arial, Helvetica, sans-serif;
                font-size: 10pt;
                line-height: 1.5;
            }
            
            h1 {
                page-break-after: avoid;
                font-size: 24pt;
                color: #2c3e50;
            }
            
            h2 {
                page-break-after: avoid;
                font-size: 18pt;
                color: #34495e;
                margin-top: 15mm;
            }
            
            h3 {
                page-break-after: avoid;
                font-size: 14pt;
                color: #2c3e50;
            }
            
            table {
                page-break-inside: avoid;
                border-collapse: collapse;
                width: 100%;
                margin: 5mm 0;
            }
            
            thead {
                background-color: #34495e !important;
                color: white !important;
            }
            
            th, td {
                padding: 3mm;
                border: 1px solid #ddd;
                text-align: left;
            }
            
            .metric-card {
                page-break-inside: avoid;
                margin: 3mm;
                padding: 5mm;
                border: 1px solid #ddd;
                border-radius: 2mm;
            }
            
            .chart-container {
                page-break-inside: avoid;
                text-align: center;
                margin: 5mm 0;
            }
            
            .chart-container img {
                max-width: 100%;
                height: auto;
            }
            
            .alert {
                page-break-inside: avoid;
                padding: 3mm;
                margin: 3mm 0;
                border-left: 3mm solid;
            }
        """
    
    def html_to_pdf(
        self,
        html_content: str,
        output_path: Union[str, Path],
        custom_css: Optional[str] = None,
        header_text: Optional[str] = None,
        footer_text: Optional[str] = None
    ) -> Path:
        """
        Convert HTML content to PDF file.
        
        Args:
            html_content: HTML string to convert
            output_path: Path to save the PDF file
            custom_css: Optional custom CSS for styling
            header_text: Optional custom header text
            footer_text: Optional custom footer text
            
        Returns:
            Path to the generated PDF file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine CSS
        css_styles = [CSS(string=self.default_print_css, font_config=self.font_config)]
        
        if custom_css:
            css_styles.append(CSS(string=custom_css, font_config=self.font_config))
        
        # Add custom header/footer if provided
        if header_text or footer_text:
            html_content = self._add_header_footer(html_content, header_text, footer_text)
        
        # Generate PDF
        HTML(string=html_content).write_pdf(
            output_path,
            stylesheets=css_styles,
            font_config=self.font_config
        )
        
        return output_path
    
    def html_file_to_pdf(
        self,
        html_path: Union[str, Path],
        output_path: Union[str, Path],
        custom_css: Optional[str] = None
    ) -> Path:
        """
        Convert HTML file to PDF.
        
        Args:
            html_path: Path to HTML file
            output_path: Path to save PDF file
            custom_css: Optional custom CSS
            
        Returns:
            Path to the generated PDF file
        """
        html_path = Path(html_path)
        html_content = html_path.read_text(encoding='utf-8')
        
        return self.html_to_pdf(html_content, output_path, custom_css)
    
    def _add_header_footer(
        self,
        html_content: str,
        header_text: Optional[str],
        footer_text: Optional[str]
    ) -> str:
        """
        Add custom header and footer to HTML content.
        
        Args:
            html_content: Original HTML content
            header_text: Header text
            footer_text: Footer text
            
        Returns:
            Modified HTML with header/footer CSS
        """
        header_css = ""
        if header_text:
            header_css = f"""
                @page {{
                    @top-center {{
                        content: "{header_text}";
                        font-family: Arial, sans-serif;
                        font-size: 10pt;
                        color: #666;
                    }}
                }}
            """
        
        footer_css = ""
        if footer_text:
            footer_css = f"""
                @page {{
                    @bottom-center {{
                        content: "{footer_text} - Page " counter(page) " of " counter(pages);
                        font-family: Arial, sans-serif;
                        font-size: 9pt;
                        color: #666;
                    }}
                }}
            """
        
        # Insert custom CSS into HTML
        custom_style = f"<style>{header_css}{footer_css}</style>"
        
        if "</head>" in html_content:
            html_content = html_content.replace("</head>", f"{custom_style}</head>")
        else:
            html_content = f"{custom_style}{html_content}"
        
        return html_content
    
    def configure_pdf_settings(
        self,
        page_size: str = "A4",
        orientation: str = "portrait",
        margins: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate CSS for custom PDF settings.
        
        Args:
            page_size: Page size (A4, Letter, Legal, etc.)
            orientation: Page orientation (portrait, landscape)
            margins: Dictionary with top, right, bottom, left margins
            
        Returns:
            CSS string with page settings
        """
        if margins is None:
            margins = {"top": "25mm", "right": "25mm", "bottom": "25mm", "left": "25mm"}
        
        css = f"""
            @page {{
                size: {page_size} {orientation};
                margin-top: {margins.get('top', '25mm')};
                margin-right: {margins.get('right', '25mm')};
                margin-bottom: {margins.get('bottom', '25mm')};
                margin-left: {margins.get('left', '25mm')};
            }}
        """
        
        return css
    
    def optimize_for_print(self, html_content: str) -> str:
        """
        Optimize HTML content for print (remove interactive elements, etc.).
        
        Args:
            html_content: Original HTML content
            
        Returns:
            Optimized HTML content
        """
        # Remove JavaScript (not supported in PDF)
        import re
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
        
        # Remove interactive elements
        html_content = re.sub(r'onclick="[^"]*"', '', html_content)
        html_content = re.sub(r'onload="[^"]*"', '', html_content)
        
        return html_content


# Example usage
if __name__ == "__main__":
    # Create sample HTML
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Report</title>
        <style>
            body { font-family: Arial; padding: 20px; }
            h1 { color: #2c3e50; }
            table { width: 100%; border-collapse: collapse; }
            th, td { border: 1px solid #ddd; padding: 8px; }
            th { background-color: #34495e; color: white; }
        </style>
    </head>
    <body>
        <h1>SynFinance Test Report</h1>
        <h2>Executive Summary</h2>
        <p>This is a test PDF export from HTML content.</p>
        
        <h3>Key Metrics</h3>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Total Transactions</td>
                    <td>1,000</td>
                </tr>
                <tr>
                    <td>Fraud Rate</td>
                    <td>2.5%</td>
                </tr>
                <tr>
                    <td>Total Amount</td>
                    <td>₹5,000,000</td>
                </tr>
            </tbody>
        </table>
    </body>
    </html>
    """
    
    # Export to PDF
    exporter = PDFExporter()
    pdf_path = exporter.html_to_pdf(
        html_content=sample_html,
        output_path="test_report.pdf",
        header_text="SynFinance Analytics Report",
        footer_text="Confidential"
    )
    
    print(f"✅ Generated PDF: {pdf_path}")
    print(f"📄 File size: {pdf_path.stat().st_size / 1024:.1f} KB")
