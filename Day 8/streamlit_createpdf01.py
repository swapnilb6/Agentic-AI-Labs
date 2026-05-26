# createpdf.py
# ==========================================================
# GENERATE SAMPLE financial_report.pdf
# ==========================================================

"""
INSTALLATION
----------------------------------------------------------

pip install fpdf2
"""

from fpdf import FPDF

# ==========================================================
# SAMPLE FINANCIAL CONTENT
# ==========================================================

financial_content = """
ACME GLOBAL BANK
ANNUAL FINANCIAL REPORT 2025

----------------------------------------------------------
EXECUTIVE SUMMARY
----------------------------------------------------------

ACME Global Bank experienced moderate revenue growth
during fiscal year 2025 despite macroeconomic challenges,
persistent inflation, and elevated global interest rates.

The institution continued focusing on:
- Digital banking expansion
- Credit risk reduction
- Liquidity optimization
- AI-driven fraud detection
- Regulatory compliance modernization

----------------------------------------------------------
MACROECONOMIC CONDITIONS
----------------------------------------------------------

The Federal Reserve increased benchmark interest rates
multiple times during 2025 to control inflation.

Higher interest rates led to:
- Increased borrowing costs
- Reduced consumer spending
- Higher mortgage defaults
- Slower corporate lending activity

Inflation remained above target levels during
the first three quarters of the year.

Economic slowdown concerns increased the probability
of recession across several sectors.

----------------------------------------------------------
REVENUE PERFORMANCE
----------------------------------------------------------

Total annual revenue increased by 4.8% compared
to the previous fiscal year.

Primary revenue contributors:
- Retail banking services
- Commercial lending
- Investment management
- Digital payment infrastructure

However, net profit margins declined due to:
- Rising operational costs
- Increased loan loss provisions
- Higher compliance expenditures

----------------------------------------------------------
CREDIT RISK ANALYSIS
----------------------------------------------------------

Loan default rates increased significantly
during the second half of 2025.

Primary drivers:
- Rising interest rates
- Consumer debt pressure
- Small business insolvencies
- Commercial real estate weakness

Credit risk exposure increased across:
- Mortgage portfolios
- SME lending
- Corporate credit lines

Risk management teams implemented stricter
underwriting standards to mitigate future losses.

----------------------------------------------------------
LIQUIDITY RISK
----------------------------------------------------------

Bank liquidity levels declined moderately
due to:
- Deposit outflows
- Market volatility
- Increased funding costs

Treasury teams increased short-term
liquidity reserves to stabilize
cash flow requirements.

Stress testing scenarios indicated
potential liquidity pressure under
severe recession conditions.

----------------------------------------------------------
FRAUD DETECTION
----------------------------------------------------------

AI-based fraud monitoring systems
detected multiple suspicious
cross-border transaction patterns.

Machine learning systems improved:
- Transaction anomaly detection
- AML monitoring
- Identity verification
- Real-time payment security

Cybersecurity investments increased
by 18% during fiscal year 2025.

----------------------------------------------------------
REGULATORY COMPLIANCE
----------------------------------------------------------

The bank continued aligning with:
- Basel III regulations
- AML compliance frameworks
- International payment security standards

Compliance costs increased due to:
- Enhanced reporting requirements
- Expanded audit procedures
- Data governance modernization

----------------------------------------------------------
SYSTEMIC FINANCIAL RISKS
----------------------------------------------------------

Systemic risks identified during 2025:
- Prolonged inflation
- Rising interest rates
- Credit market instability
- Liquidity shortages
- Global recession risk

Management believes elevated interest
rates may continue increasing:
- Loan defaults
- Credit exposure
- Funding stress

These conditions could negatively impact
bank liquidity and profitability.

----------------------------------------------------------
OUTLOOK FOR 2026
----------------------------------------------------------

The bank expects:
- Slower lending growth
- Increased digital banking adoption
- Continued regulatory scrutiny
- Higher AI investment
- Ongoing risk management modernization

Strategic priorities include:
- AI-powered financial intelligence
- Hybrid cloud infrastructure
- Fraud prevention automation
- Customer experience optimization

----------------------------------------------------------
END OF REPORT
----------------------------------------------------------
"""

# ==========================================================
# CREATE PDF
# ==========================================================

pdf = FPDF()

pdf.set_auto_page_break(auto=True, margin=15)

pdf.add_page()

# ==========================================================
# TITLE
# ==========================================================

pdf.set_font("Helvetica", "B", 18)

pdf.cell(
    200,
    10,
    txt="ACME GLOBAL BANK - FINANCIAL REPORT 2025",
    ln=True,
    align="C"
)

pdf.ln(10)

# ==========================================================
# BODY
# ==========================================================

pdf.set_font("Helvetica", size=12)

pdf.multi_cell(
    0,
    8,
    financial_content
)

# ==========================================================
# SAVE PDF
# ==========================================================

output_file = "financial_report.pdf"

pdf.output(output_file)

print(f"\nPDF successfully created: {output_file}")