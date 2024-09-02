# SMI Momentum Breakout Strategy Backtesting

This repository contains a Python-based implementation of the SMI Momentum Breakout Strategy, including enhancements such as volume filters, divergence checks, ATR-based stop losses, and profit targets. The strategy is tested using the `backtesting.py` library and is optimized with multiprocessing for efficient backtesting of various parameter combinations.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [License](#license)

## Project Overview

The SMI Momentum Breakout Strategy is a comprehensive momentum breakout trading strategy that leverages a combination of technical indicators to identify high-probability entry points. This project is designed for traders who are looking to capitalize on strong market moves signaled by momentum shifts.

## Features

- **Exponential Moving Average (EMA)**: Determines the overall trend direction.
- **Bollinger Bands**: Measures market volatility and identifies potential breakouts.
- **Stochastic Momentum Index (SMI)**: Provides insight into the direction of the breakout.
- **Volume Filter**: Validates the strength of breakouts by checking trading volume.
- **Divergence Check**: Detects potential reversals by comparing price and SMI divergence.
- **ATR-Based Stop Loss**: Dynamically adjusts stop losses based on market volatility.
- **Profit Targets**: Sets profit targets based on key support and resistance levels.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   `git clone https://github.com/nashmutazu/smi-momentum-breakout.git`
   `cd smi-momentum-breakout`

2. **Create and activate a virtual environment (optional but recommended):**
    ```bash
    `python -m venv env`
    `source env/bin/activate`


3. **Install the required dependencies:**
    ```bash
    `pip install -r requirements.txt`
Ensure your requirements.txt includes necessary libraries like backtesting, pandas, and numpy.

## Usage
1. **Prepare Your Data:**

- Prepare your historical market data and save it as a CSV file (e.g., `your_data.csv`). The CSV file should have the following columns: `Time`, `Volume`, `Open`, `Low`, `Close`, `High`..
- Save this DataFrame as df in the script or modify the script to load your data.

2. **Run the Backtest:**
- The main script runs multiple backtests with different combinations of strategy parameters using multiprocessing. To execute the backtests, simply run the script:
    
    ```bash
    `python SMIMomentumBreakoutBacktest.py`

3. **Update output data file:** 
Specify the path to your data file and the output file for results:
    ```python
    df = pd.read_csv('your_data.csv')
    results_file = 'backtest_results.csv'
    ```

4. **Analyze the results:e:**  Analyze the results:
    The results of each configuration will be saved to `backtest_results.csv`. You can analyze this file using any spreadsheet software or load it into a DataFrame for further analysis.

### Requirements

- `backtesting.py`: A Python library for backtesting trading strategies.
- `pandas`: Used for data manipulation and analysis.
- `multiprocessing`: Part of the Python standard library, used for parallel processing.


## Configuration
The strategy can be customized using the following considerations, which can be toggled on or off:

- consider_volume_filter: Validates the strength of breakouts with volume analysis.
- consider_divergence: Checks for divergence between price and SMI before a breakout.
- consider_atr_stop_loss: Uses ATR to dynamically adjust stop losses.
- consider_profit_targets: Sets profit targets based on support and resistance levels.
These parameters are automatically configured as different combinations and tested in parallel.

## Output
- CSV File: The results of all backtest combinations are stored in backtest_results.csv.
- Console Output: The script prints the results for each parameter combination.
- Plots: The script generates a plot for the final backtest scenario.

## License
This project is licensed under the MIT License. See the LICENSE file for more information.

    ```bash
    `This code block represents the content of your 'README.md' file, formatted in Markdown. You can copy and paste it directly into your 'README.md' file in your repository.`
