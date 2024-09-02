import os

import csv
from tqdm import tqdm
import multiprocessing as mp
import pandas_ta as ta
from backtesting import Backtest, Strategy
from itertools import product
import pandas as pd
from backtesting.lib import crossover
import datetime as dt
import json

# Assuming the same imports and strategy definition from before
FILE_NAME = "SMIMomentumBreakout"
FOLDER_NAME = "discovery_testing"

pairs = [
    "SPX500_USD",
    "UK100_GBP",
    "DE30_EUR",
    "XAU_USD",
    "BCO_USD",
    "CORN_USD",
    "XCU_USD",
    "US2000_USD",
    "XAG_USD",
    "GBP_JPY",
    "EUR_USD",
    "GBP_USD",
    "EUR_USD",
]

granualrity = ["M15", "M30", "H1"]


def isNowInTimePeriod(startTime, endTime, nowTime):
    if startTime < endTime:
        return nowTime >= startTime and nowTime <= endTime
    else:
        # Over midnight:
        return nowTime >= startTime or nowTime <= endTime


def construct_df_O(
    commodity, granualrity, min_year=2023, max_year=None, DIRECTORY_PATH="./data"
):
    if os.path.isdir(DIRECTORY_PATH):
        path = DIRECTORY_PATH
    elif os.path.isdir("./data"):
        path = "./data"
    elif os.path.isdir("../data"):
        path = "../data"
    else:
        path = "../../data"

    df = pd.read_pickle(f"{path}/{commodity}_{granualrity}.pkl")

    df.rename(
        columns={
            "time": "Time",
            "volume": "Volume",
            "mid_o": "Open",
            "mid_h": "High",
            "mid_c": "Close",
            "mid_l": "Low",
            "bid_o": "B_Open",
            "bid_h": "B_High",
            "bid_c": "B_Close",
            "bid_l": "B_Low",
            "ask_o": "A_Open",
            "ask_h": "A_High",
            "ask_c": "A_Close",
            "ask_l": "A_Low",
        },
        inplace=True,
    )

    df["Time"] = pd.to_datetime(df["Time"])

    df = df[
        (df["Time"] > str(min_year))
        & (df["Time"] < str(min_year + 1 if not max_year else max_year))
    ]

    df = df[["Time", "Volume", "High", "Close", "Open", "Low"]]
    df.reset_index(drop=True, inplace=True)

    return df


# Placeholder function to calculate the 200-period SMA
def calculate_sma(data, period=200):
    return ta.sma(pd.Series(data), period)


# Placeholder function to calculate the 200-period EMA
def calculate_ema(data, period=200):
    return ta.ema(pd.Series(data), period)


def get_bbands(price, bb_length, bb_std, key):
    bbands = ta.bbands(pd.Series(price), bb_length, bb_std)

    formated_std = "{:.1f}".format(bb_std)

    bbands.rename(
        columns={
            f"BBL_{bb_length}_{formated_std}": "BBL",
            f"BBM_{bb_length}_{formated_std}": "BBM",
            f"BBU_{bb_length}_{formated_std}": "BBU",
        },
        inplace=True,
    )

    return bbands[[key]]


def ema(series, period):
    return calculate_ema(series, period)


def get_smi(high, low, close, N=14, M=3, signal_period=3):
    HLM = (pd.Series(high).rolling(N).max() + pd.Series(low).rolling(N).min()) / 2
    D = close - HLM
    DS = ema(ema(D, M), M)
    SHL = ema(
        ema(pd.Series(high).rolling(N).max() - pd.Series(low).rolling(N).min(), M), M
    )
    SMI = (DS / (0.5 * SHL)) * 100
    return SMI


def get_volume_average(series, length):
    return pd.Series(series).rolling(length).mean()


def get_atr(high, low, close, length):
    return ta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length)


# Define the strategy
class SMIMomentumBreakout(Strategy):
    units = 0.1
    ema = 500
    sma = 10
    bb_length = 20
    bb_std = 2
    bb_squeeze = 0.02
    smi_length = 14
    smi_fast = 3
    signal_period = 3
    volume_average_length = 20
    atr_length = 14
    fixed_profit_target = 2  # Example: 2x initial risk
    trailing_stop_multiplier = 1.5  # Ex
    year = 2024

    def init(self):
        self.ema_500 = self.I(calculate_ema, self.data.Close, self.ema)
        self.sma_10 = self.I(calculate_sma, self.data.Close, self.sma)
        self.upper_band = self.I(
            get_bbands, self.data.Close, self.bb_length, self.bb_std, "BBU"
        )
        self.lower_band = self.I(
            get_bbands, self.data.Close, self.bb_length, self.bb_std, "BBL"
        )
        self.smi = self.I(
            get_smi,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.smi_length,
            self.smi_fast,
            self.smi_fast,
        )
        self.signal_line = self.I(calculate_ema, self.smi, self.smi_fast)
        self.volume_avg = self.I(
            get_volume_average, self.data.Volume, self.volume_average_length
        )
        self.atr = self.I(
            get_atr, self.data.High, self.data.Low, self.data.Close, self.atr_length
        )

    def next(self):
        if len(self.smi) < 2:
            return

        date_time = pd.to_datetime(self.data.Time[-1])
        day = date_time.day
        month = date_time.month
        year = date_time.year
        hour = date_time.hour
        minute = date_time.minute

        can_trade = isNowInTimePeriod(
            dt.time(9, 0), dt.time(21, 0), dt.time(hour, minute)
        )

        if year < self.year:
            return

        # Long Entry Conditions
        if (
            self.data.Close[-1] > self.ema_500[-1]
            and self.data.Close[-1] > self.sma_10[-1]
            and can_trade
        ):
            if (
                self.upper_band[-1] - self.lower_band[-1]
                < self.bb_squeeze * self.sma_10[-1]
            ):
                if (
                    crossover(self.smi, self.signal_line)
                    and self.data.Close[-1] > self.upper_band[-1]
                ):
                    if self.data.Volume[-1] >= self.volume_avg[-1]:  # Volume Filter
                        sl = self.data.Low[-3:].min()
                        risk = self.data.Close[-1] - sl
                        profit_target = (
                            self.data.Close[-1] + self.fixed_profit_target * risk
                        )

                        try:
                            self.buy(tp=profit_target, size=self.units)
                        except Exception as e:
                            pass

        # Short Entry Conditions
        elif (
            self.data.Close[-1] < self.ema_500[-1]
            and self.data.Close[-1] < self.sma_10[-1]
            and can_trade
        ):
            if (
                self.upper_band[-1] - self.lower_band[-1]
                < self.bb_squeeze * self.sma_10[-1]
            ):
                if (
                    crossover(self.signal_line, self.smi)
                    and self.data.Close[-1] < self.lower_band[-1]
                ):
                    if self.data.Volume[-1] >= self.volume_avg[-1]:  # Volume Filter
                        sl = self.data.High[-3:].max()
                        risk = sl - self.data.Close[-1]
                        profit_target = (
                            self.data.Close[-1] - self.fixed_profit_target * risk
                        )

                        try:
                            self.sell(tp=profit_target, size=self.units)
                        except Exception as e:
                            pass

        for index in range(len(self.trades)):
            position = self.trades[index]
            # Trailing Stop-Loss for Long Positions
            if position.is_long:
                if position.sl == None:
                    position.sl = (
                        self.data.Close[-1]
                        - self.trailing_stop_multiplier * self.atr[-1]
                    )

                position.sl = max(
                    position.sl,
                    self.data.Close[-1] - self.trailing_stop_multiplier * self.atr[-1],
                )
                if self.data.Close[-1] < self.sma_10[-1] or crossover(
                    self.signal_line, self.smi
                ):
                    position.close()

            # Trailing Stop-Loss for Short Positions
            if position.is_short:
                if position.sl == None:
                    position.sl = (
                        self.data.Close[-1]
                        + self.trailing_stop_multiplier * self.atr[-1]
                    )

                position.sl = min(
                    position.sl,
                    self.data.Close[-1] + self.trailing_stop_multiplier * self.atr[-1],
                )
                if self.data.Close[-1] > self.sma_10[-1] or crossover(
                    self.smi, self.signal_line
                ):
                    position.close()


def run_forward_test_and_save(params):
    pair, granualrity = params

    print(f"** Running {FILE_NAME} script for {pair} **")
    optimized_results = get_optimized_results()
    
    pair_df = optimized_results[optimized_results["pair"] == pair]
    
    pair_df.reset_index(inplace=True, drop=True)
    
    df_forward = construct_df_O(pair, granualrity, 2023, 2025)

    for index in range(len(pair_df)):
        param_dict = pair_df.iloc[index]

        equity_final = param_dict["Equity Final[$]"]

        if equity_final < 5000:
            continue

        best_params = {
            "ema": param_dict["ema"],
            "sma": param_dict["sma"],
            "bb_length": param_dict["bb_length"],
            "bb_std": param_dict["bb_std"],
            "bb_squeeze": param_dict["bb_squeeze"],
            "smi_length": param_dict["smi_length"],
            "smi_fast": param_dict["smi_fast"],
            "volume_average_length": param_dict["volume_average_length"],
            "atr_length": param_dict["atr_length"],
            "fixed_profit_target": param_dict["fixed_profit_target"],
            "trailing_stop_multiplier": param_dict["trailing_stop_multiplier"],
        }

        bt = Backtest(
            df_forward, SMIMomentumBreakout, cash=5000, commission=0.0002, margin=1 / 30
        )
        stats = bt.run(**best_params)


        if stats["Equity Final [$]"] > 5000:
            result = [
                pair,
                granualrity,
                param_dict["ema"],
                param_dict["sma"],
                param_dict["bb_length"],
                param_dict["bb_std"],
                param_dict["bb_squeeze"],
                param_dict["smi_length"],
                param_dict["smi_fast"],
                param_dict["volume_average_length"],
                param_dict["atr_length"],
                param_dict["fixed_profit_target"],
                param_dict["trailing_stop_multiplier"],
                stats["Return [%]"],
                stats["Equity Final [$]"],
                stats["Sharpe Ratio"],
                stats["Max. Drawdown [%]"],
                stats["Win Rate [%]"],
                stats["# Trades"],
            ]

            # Save the result immediately after the backtest
            save_result_to_csv(result)


def get_optimized_results():
    csv_file = FILE_NAME + "_" + f"optimization_results.csv"
    optimized_results = pd.read_csv(csv_file)

    return optimized_results


# Function to save a single result to CSV
def save_result_to_csv(result):
    csv_file = FILE_NAME + "_" + f"forward_test_results.csv"

    write_header = not pd.io.common.file_exists(csv_file)
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(
                [
                    "pair",
                    "granularity",
                    "ema",
                    "sma",
                    "bb_length",
                    "bb_std",
                    "bb_squeeze",
                    "smi_length",
                    "smi_fast",
                    "volume_average_length",
                    "atr_length",
                    "fixed_profit_target",
                    "trailing_stop_multiplier",
                    "Return [%]",
                    "Equity Final [$]",
                    "Sharpe Ratio",
                    "Max. Drawdown [%]",
                    "Win Rate [%]",
                    "Total Trades",
                ]
            )
        writer.writerow(result)


# Main function to set up multiprocessing
def main():
    print("** Starting **")
    trade_combinations = list(product(pairs, granualrity))

    pool = mp.Pool(mp.cpu_count())  # Use all available cores
    for _ in tqdm(
        pool.imap_unordered(run_forward_test_and_save, trade_combinations),
        total=len(pairs),
    ):
        pass  # The progress bar will update with each completed task

    pool.close()
    pool.join()

    print(f"Optimization completed. Results saved.")


# Use multiprocessing to run the backtests in parallel
if __name__ == "__main__":
    main()
