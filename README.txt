#Final Project: Quantitative Analysis & Linux/Git Workflow

This project constitutes the final submission for the "Python, Git & Linux" module. It combines an in-depth financial analysis of a crypto-asset (Bitcoin) with software development and version control best practices.


#Table of Contents
1. [Project Context](#-project-context)
2. [Part A: Quantitative Analysis (Python)](#-part-a-quantitative-analysis-python)
    - [Data Retrieval](#1-data-retrieval)
    - [Trading Strategies & Backtesting](#2-trading-strategies--backtesting)
    - [Visualization](#3-visualization)
3. [Part B: Environment & Workflow (Git/Linux)](#-part-b-environment--workflow-gitlinux)
4. [Installation and Prerequisites](#-installation-and-prerequisites)
5. [Disclaimer](#-disclaimer)


#Project Context

The objective is to develop a complete financial data processing pipeline:
* API Polling to obtain real-time/historical market data.
* Data Cleaning & Structuring using Pandas.
* Algorithm Implementation for trading (Momentum vs. Buy & Hold).
* Code Management via Git and execution within a Linux environment.


#Part A: Quantitative Analysis (Python)

This section focuses on the Bitcoin (BTC-USD) asset. The code is structured in the `Python_git_linux_final.ipynb` notebook.

# 1. Data Retrieval
* Source: [Alpha Vantage](https://www.alphavantage.co/) API.
* Method: HTTP requests via the `requests` library.
* Processing: 
    * Time series extraction (`DIGITAL_CURRENCY_DAILY`).
    * Conversion to a `pandas` DataFrame.
    * Type cleaning (float) and time indexing (`datetime`).

# 2. Trading Strategies & Backtesting
Two investment strategies are compared over the historical period:

# a. "Buy & Hold" Strategy (Passive)
* Logic: Buy on the first available day and hold until the last day.
* Formula: $(Price_{Final} - Price_{Initial}) / Price_{Initial}$
* Observed Performance: ~182,862,860% (Strong historical appreciation of BTC).

# b. "Momentum" Strategy (Active)
* Indicators: Moving Averages (MA).
    * MA20 (Short term).
    * MA50 (Medium term).
* Signal: 
    * Buy (1): When the MA20 crosses above the MA50 (Golden Cross).
    * Sell/Cash (0): When the MA20 crosses below the MA50.
* Observed Performance: ~278,306,339%.
* Analysis: The trend-following strategy outperformed simple "Buy & Hold" over this specific period by avoiding certain major downturns while capturing the asset's upside volatility.

# 3. Visualization
Use of `matplotlib` to generate:
* The BTC price curve ("Closing Price").
* Overlay of moving averages (MA20 and MA50).
* Visual markers for Buy (▲ Green) and Sell (▼ Red) signals.

---

# 🐧 Part B: Environment & Workflow (Git/Linux)

This section validates mastery of the development environment.

# Version Control (Git)
The project adheres to versioning standards:
* Initialization: Locally configured Git repository.
* Commits: Clear history of modifications (adding API, JSON parsing fix, adding charts).
* Exclusion: Use of a `.gitignore` to exclude temporary files (`.ipynb_checkpoints`, `__pycache__`) and sensitive configuration files.

# Linux Environment
The script is designed to be executable in a Linux environment (Ubuntu/Debian):
1.  Interpreter: Python 3.x.
2.  Virtual Environment: Dependency isolation recommended.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Automation (Optional): This notebook can be converted into a Python script (`.py`) to be executed via a scheduled task (`cron`) to fetch data daily at 8:00 AM.

---

# 🛠 Installation and Prerequisites

# Dependencies
The project requires the following Python libraries (listed in `requirements.txt`):
```text
pandas
matplotlib
requests
