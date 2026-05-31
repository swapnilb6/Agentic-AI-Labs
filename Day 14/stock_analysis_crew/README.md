# 📈 Stock Profile Analysis — CrewAI Multi-Agent System

A production-ready multi-agent AI application that performs comprehensive stock profile analysis
using the **CrewAI 1.10+** framework, compatible with **Python 3.10–3.13**.

---

## 🏗️ Architecture Overview

```
stock_analysis_crew/
├── agents/
│   ├── __init__.py
│   ├── data_collector.py       # Fetches raw stock & financial data
│   ├── fundamental_analyst.py  # Analyses financials & valuation
│   ├── technical_analyst.py    # Price action & momentum analysis
│   ├── sentiment_analyst.py    # News & market sentiment
│   └── report_writer.py        # Synthesises & writes final report
├── tasks/
│   ├── __init__.py
│   └── analysis_tasks.py       # Task definitions for each agent
├── tools/
│   ├── __init__.py
│   ├── stock_data_tool.py      # yfinance wrapper tool
│   ├── financial_metrics_tool.py # Key ratio calculations
│   └── news_search_tool.py     # News fetching tool
├── config/
│   ├── __init__.py
│   └── settings.py             # Centralised configuration
├── reports/                    # Generated reports saved here
├── tests/
│   └── test_tools.py           # Unit tests
├── main.py                     # Entry point
├── crew.py                     # Crew assembly & orchestration
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🤖 Agents & Their Roles

| Agent | Role | Responsibility |
|-------|------|----------------|
| **Data Collector** | Market Data Specialist | Fetches price history, volume, and company info |
| **Fundamental Analyst** | Financial Expert | P/E, EPS, revenue growth, debt ratios |
| **Technical Analyst** | Chart Specialist | Moving averages, RSI, MACD, support/resistance |
| **Sentiment Analyst** | News & Sentiment Researcher | News sentiment, market perception |
| **Report Writer** | Investment Strategist | Synthesises all findings into a structured report |

---

## ⚙️ Requirements

- **Python**: 3.10 – 3.13
- **OS**: macOS, Linux, Windows
- **LLM Provider**: OpenAI (default) or Anthropic Claude

---

## 🚀 Setup & Installation

### 1. Clone / download the project

```bash
cd stock_analysis_crew
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 5. Run the analysis

```bash
# Interactive mode (prompts for ticker)
python main.py

# Direct mode
python main.py --ticker AAPL

# With custom output file
python main.py --ticker MSFT --output reports/msft_report.md

# Using Anthropic Claude instead of OpenAI
python main.py --ticker NVDA --provider anthropic
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | If using OpenAI | Your OpenAI API key |
| `ANTHROPIC_API_KEY` | If using Anthropic | Your Anthropic API key |
| `LLM_PROVIDER` | No (default: openai) | `openai` or `anthropic` |
| `LLM_MODEL` | No | Override default model |
| `MAX_TOKENS` | No (default: 4096) | Max tokens per response |
| `TEMPERATURE` | No (default: 0.1) | LLM temperature |

---

## 📊 Sample Output

The system generates a Markdown report covering:
- Company overview & sector
- Fundamental analysis (P/E, PEG, EV/EBITDA, margins)
- Technical analysis (trend, momentum, key levels)
- Sentiment & news summary
- Risk factors
- Investment thesis & recommendation

---

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**.  
It does **not** constitute financial advice. Always do your own research  
and consult a qualified financial advisor before making investment decisions.
