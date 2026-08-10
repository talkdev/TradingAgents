import os
import sys
import glob
import re
import sqlite3
import threading
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ==========================================
# 1. Base Configuration Variables (Linux)
# ==========================================
MAX_WORKERS = 6  # Process exactly 6 stocks in parallel at a time

BASE_DIR = "/root/TradingAI"
TRADINGAGENTS_PATH = os.path.join(BASE_DIR, "TradingAgents")
BASE_OUTPUT_FOLDER = os.path.join(TRADINGAGENTS_PATH, "reports")
STOCKS_FILE = os.path.join(TRADINGAGENTS_PATH, "stocks.txt")
DB_PATH = os.path.join(BASE_OUTPUT_FOLDER, "trading_history.db")

if TRADINGAGENTS_PATH not in sys.path:
    sys.path.append(TRADINGAGENTS_PATH)

# Set local dummy API key for local vLLM instance
os.environ["OPENAI_API_KEY"] = "local-dummy-key"

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Global thread lock for safe database operations
db_lock = threading.Lock()


# ==========================================
# 2. Database Management Functions
# ==========================================
def init_db(db_path, tickers):
    """
    Initializes SQLite DB with run_date as rows and stock names as columns.
    Dynamically adds columns if new tickers appear in stocks.txt.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_signals (
            run_date TEXT PRIMARY KEY
        )
    """)
    conn.commit()

    cursor.execute("PRAGMA table_info(stock_signals)")
    existing_cols = {col[1] for col in cursor.fetchall()}

    for ticker in tickers:
        col_name = ticker  # Retain full ticker name (e.g., "RELIANCE.NS")
        if col_name not in existing_cols:
            cursor.execute(f'ALTER TABLE stock_signals ADD COLUMN "{col_name}" TEXT')
            print(f"[+] Added new column to DB: {col_name}")

    conn.commit()
    conn.close()


def record_run_to_db(db_path, target_date, status_dict):
    """
    Inserts or updates the run_date row with stock stances for the date.
    """
    with db_lock:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("INSERT OR IGNORE INTO stock_signals (run_date) VALUES (?)", (target_date,))
        
        for ticker, stance in status_dict.items():
            query = f'UPDATE stock_signals SET "{ticker}" = ? WHERE run_date = ?'
            cursor.execute(query, (stance, target_date))

        conn.commit()
        conn.close()
        print(f"[✔] Successfully recorded execution statuses in local DB for {target_date}")


def _markdown_table_cell(value) -> str:
    """Return a safe, single-line Markdown table cell value."""
    return str(value or "").replace("\\", "\\\\").replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def generate_change_report(db_path, target_date, output_folder):
    """
    Queries DB for status changes compared to the previous run date,
    and writes the summary-<date>.md report in base reports folder.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT run_date FROM stock_signals ORDER BY run_date DESC LIMIT 2")
    dates = [row[0] for row in cursor.fetchall()]

    summary_file_path = os.path.join(output_folder, f"summary-{target_date}.md")

    if len(dates) < 2:
        report_content = (
            f"# Trading Signal Change Summary — {target_date}\n\n"
            "**Note:** This is the first recorded run in the local database. "
            "No previous run data is available for comparison.\n\n"
            "### Current Statuses Recorded:\n"
        )
        cursor.execute("PRAGMA table_info(stock_signals)")
        cols = [col[1] for col in cursor.fetchall() if col[1] != "run_date"]
        
        cursor.execute("SELECT * FROM stock_signals WHERE run_date = ?", (target_date,))
        row = cursor.fetchone()
        
        if row:
            col_names = [description[0] for description in cursor.description]
            val_map = dict(zip(col_names, row))
            for ticker in cols:
                status = val_map.get(ticker, "N/A")
                if status:
                    report_content += f"- **{ticker}:** `{status}`\n"

        with open(summary_file_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"[✔] First run summary written to: {summary_file_path}")
        conn.close()
        return

    current_date, prev_date = dates[0], dates[1]

    cursor.execute("SELECT * FROM stock_signals WHERE run_date = ?", (current_date,))
    curr_row = dict(zip([d[0] for d in cursor.description], cursor.fetchone()))

    cursor.execute("SELECT * FROM stock_signals WHERE run_date = ?", (prev_date,))
    prev_row = dict(zip([d[0] for d in cursor.description], cursor.fetchone()))

    conn.close()

    changes = []
    unchanged = []

    for col in curr_row:
        if col == "run_date":
            continue
        curr_val = curr_row.get(col)
        prev_val = prev_row.get(col)

        if curr_val and curr_val != prev_val:
            changes.append((col, prev_val or "UNKNOWN/NEW", curr_val))
        elif curr_val:
            unchanged.append((col, curr_val))

    report_md = f"# Trading Signal Change Report — {current_date}\n"
    report_md += f"**Comparison Window:** `{prev_date}` → `{current_date}`\n\n"
    report_md += "---\n\n"

    report_md += "## 🚨 Stocks with Status Changes\n\n"
    if changes:
        report_md += "| Stock Symbol | Previous Stance (`" + prev_date + "`) | New Stance (`" + current_date + "`) |\n"
        report_md += "| :--- | :--- | :--- |\n"
        for ticker, p_val, c_val in sorted(changes):
            report_md += (
                f"| `{_markdown_table_cell(ticker)}` | "
                f"`{_markdown_table_cell(p_val)}` | "
                f"`{_markdown_table_cell(c_val)}` |\n"
            )
    else:
        report_md += "*No stocks changed status compared to the previous run.*\n"

    report_md += "\n---\n\n"
    report_md += "## 📋 Unchanged / Stable Positions\n\n"
    if unchanged:
        report_md += "| Stock Symbol | Sustained Stance |\n"
        report_md += "| :--- | :--- |\n"
        for ticker, val in sorted(unchanged):
            report_md += f"| `{_markdown_table_cell(ticker)}` | `{_markdown_table_cell(val)}` |\n"
    else:
        report_md += "*No stocks retained the same status compared to the previous run.*\n"

    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[✔] Status Change Report successfully written to: {summary_file_path}")


# ==========================================
# 3. Auxiliary Functions
# ==========================================
def load_tickers_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"[!] Warning: Ticker file '{file_path}' not found.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_matches = re.findall(r'\b[A-Z0-9&\-]+\.NS\b|\b[A-Z0-9&\-]+\b', content)

    cleaned_tickers = []
    for symbol in raw_matches:
        symbol = symbol.strip()
        if not symbol.endswith(".NS") and symbol.isupper():
            symbol = f"{symbol}.NS"
        if len(symbol) > 3 and symbol.endswith(".NS"):
            cleaned_tickers.append(symbol)

    seen = set()
    return [t for t in cleaned_tickers if not (t in seen or seen.add(t))]


def parse_stance_from_summary(summary_path):
    if not os.path.exists(summary_path):
        return "UNKNOWN"

    with open(summary_path, "r", encoding="utf-8") as f:
        text = f.read()

    match = re.search(r"-\s*\*\*Stance:\*\*\s*\[?([^\]\n]+)\]?", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().upper()

    if "BUY" in text.upper():
        return "BUY"
    elif "SHORT" in text.upper() or "SELL" in text.upper():
        return "SELL"
    elif "HOLD" in text.upper():
        return "HOLD"
    
    return "NO TRADE"


def generate_ai_summary(ticker, target_date, report_dir):
    print(f"[+] Generating Swing Summary for {ticker} ({target_date})...")
    
    md_files = glob.glob(os.path.join(report_dir, "*.md"))
    combined_reports_text = ""
    for file_path in sorted(md_files):
        if "summary.md" in os.path.basename(file_path).lower():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            file_name = os.path.basename(file_path)
            combined_reports_text += f"\n\n--- Start of {file_name} ---\n" + f.read() + f"\n--- End of {file_name} ---\n"
            
    if not combined_reports_text.strip():
        print(f"[!] No raw reports found in {report_dir} to summarize.")
        return "UNKNOWN"

    # Connect to local vLLM server
    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="local-dummy-key"
    )

    system_prompt = (
        "You are a Lead Quantitative Swing Trader. Evaluate this stock strictly as a SHORT-TERM SWING TRADE setup (5 to 20 days).\n\n"
        "Your output MUST follow this exact Markdown structure:\n\n"
        "### 1. Swing Trade Decision\n"
        "- **Stance:** [BUY / SHORT / NO TRADE (WAIT FOR BREAKOUT) / HOLD]\n"
        "- **Target Time Horizon:** [E.g., 5 to 15 Trading Days]\n"
        "- **Setup Pattern:** [E.g., Pullback to 50 SMA / Range Breakout]\n\n"
        "### 2. Execution Parameters\n"
        "- **Trigger / Entry Zone:** [Price]\n"
        "- **Take Profit Target 1 (TP1):** [Price]\n"
        "- **Take Profit Target 2 (TP2):** [Price]\n"
        "- **Stop Loss:** [Price]\n"
        "- **Risk-to-Reward Ratio (R:R):** [Ratio]\n"
        "- **Position Sizing:** [Percentage]\n\n"
        "### 3. Key Technical & Catalyst Drivers\n"
        "- Bullet points of catalysts.\n\n"
        "### 4. Setup Invalidation Conditions\n"
        "- Invalidation details.\n"
    )
    
    try:
        response = client.chat.completions.create(
            model="Qwen2.5-72B-Instruct-AWQ",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_reports_text}
            ],
            temperature=0.1,
        )
        
        summary_content = response.choices[0].message.content
        summary_file_path = os.path.join(report_dir, "summary.md")
        
        with open(summary_file_path, "w", encoding="utf-8") as f:
            f.write(summary_content)
            
        print(f"[✔] Swing Trade Summary saved to: {summary_file_path}")
        return parse_stance_from_summary(summary_file_path)
        
    except Exception as e:
        print(f"[!] Failed to generate local Qwen summary for {ticker}: {str(e)}")
        return "ERROR"


def process_single_stock(ticker, target_date, base_output_folder, config):
    report_dir = os.path.join(base_output_folder, ticker, target_date)
    os.makedirs(report_dir, exist_ok=True)

    print(f"\n[🚀 RUNNING] Started analysis for {ticker} -> {report_dir}")
    
    try:
        ta = TradingAgentsGraph(debug=False, config=config)
        state, decision = ta.propagate(ticker, target_date)
        
        reports_to_save = {
            "00_Portfolio_Manager_Decision.md": decision,
            "01_Fundamentals_Report.md": state.get("fundamentals_report", ""),
            "02_Market_Technical_Report.md": state.get("market_report", ""),
            "03_News_Macro_Report.md": state.get("news_report", ""),
            "04_Sentiment_Report.md": state.get("sentiment_report", "")
        }
        
        for filename, content in reports_to_save.items():
            if content:
                if not isinstance(content, str):
                    try:
                        content = content.content
                    except AttributeError:
                        content = str(content)
                        
                file_path = os.path.join(report_dir, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
        
        stance = generate_ai_summary(ticker, target_date, report_dir)
        print(f"[✔ COMPLETED] Finished {ticker} | Final Stance: {stance}")
        return ticker, stance

    except Exception as e:
        print(f"[!] Error processing {ticker}: {str(e)}")
        return ticker, "ERROR"


# ==========================================
# 4. Main Execution Engine
# ==========================================
def analyze_stocks():
    # The default contains nested dictionaries; take a deep copy so this
    # runner's optional-source policy cannot leak into another graph instance.
    config = deepcopy(DEFAULT_CONFIG)
    os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)
    
    # Local vLLM Configuration
    config["llm_provider"] = "openai_compatible"
    config["backend_url"] = "http://127.0.0.1:8000/v1"
    config["deep_think_llm"] = "Qwen2.5-72B-Instruct-AWQ"
    config["quick_think_llm"] = "Qwen2.5-72B-Instruct-AWQ"
    
    config["max_debate_rounds"] = 2
    config["max_risk_discuss_rounds"] = 1
    
    # These public/optional sources are not dependable for a parallel NSE
    # batch: FRED requires a key, Polymarket can be DNS-blocked, StockTwits
    # lacks many NSE cashtags, and Reddit throttles anonymous RSS traffic.
    # Disable them explicitly instead of issuing failing calls for every stock.
    config["data_vendors"]["macro_data"] = "disabled"
    config["data_vendors"]["prediction_markets"] = "disabled"
    config["social_sources"] = {"stocktwits": False, "reddit": False}
    
    # India-specific queries
    config["global_news_queries"] = [
        "RBI Reserve Bank of India interest rates inflation",
        "Nifty 50 Sensex Indian economy GDP",
        "FII DII institutional flows Indian markets",
        "SEBI regulations Indian stock market"
    ]

    tickers = load_tickers_from_file(STOCKS_FILE)
    if not tickers:
        print(f"[!] No valid tickers found in '{STOCKS_FILE}'. Exiting.")
        return

    # Use India Time (Asia/Kolkata) to prevent UTC date mismatches on cloud servers
    target_date = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")

    # Step 1: Initialize DB schema dynamically for all loaded tickers
    init_db(DB_PATH, tickers)

    print(f"\n{'='*70}")
    print(f"  PARALLEL PROCESSING ENGINE STARTED")
    print(f"  Target Analysis Date: {target_date}")
    print(f"  Total Tickers Loaded: {len(tickers)}")
    print(f"  Parallel Concurrency: {MAX_WORKERS} Workers")
    print(f"  Base Output Directory: {BASE_OUTPUT_FOLDER}")
    print(f"  Local History Database: {DB_PATH}")
    print(f"{'='*70}\n")

    status_results = {}

    # Step 2: Run parallel execution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_stock, ticker, target_date, BASE_OUTPUT_FOLDER, config): ticker
            for ticker in tickers
        }

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result_ticker, stance = future.result()
                status_results[result_ticker] = stance
            except Exception as exc:
                print(f"[!] Thread Exception for {ticker}: {exc}")
                status_results[ticker] = "ERROR"

    # Step 3: Record today's results into SQLite DB
    record_run_to_db(DB_PATH, target_date, status_results)

    # Step 4: Compare against previous run date and generate summary-<date>.md
    generate_change_report(DB_PATH, target_date, BASE_OUTPUT_FOLDER)

if __name__ == "__main__":
    analyze_stocks()
