import math
import scipy
import numpy as np
import pandas as pd
import gymnasium as gym
import control
import statsmodels.api as sm
import os
import itertools
import sys

from argparse import ArgumentParser
from finrl import config
from finrl.config import(

#data and model here we choose DQN
DATA_SAVE_DIR,
TRAINED_MODEL_DIR,
TENSORBOARD_LOG_DIR,
RESULTS_DIR,
INDICATORS,
TRAIN_START_DATE,
TRAIN_END_DATE,
TEST_START_DATE,
TEST_END_DATE,
TRADE_START_DATE,
TRADE_END_DATE,
ERL_PARAMS,
RLlib_PARAMS,

)



def place_market_order(side: str, ticker: str, quantity: float, price: float) -> bool:
    """Place a market order - DO NOT MODIFY

    Parameters
    ----------
    side
        Side of order to place ("BUY" or "SELL")
    ticker
        Ticker of order to place ("A", "B", or "C")
    quantity
        Volume of order to place
    price
        Price of order to place

    Returns
    -------
    True if order succeeded, False if order failed due to rate limiting

    ((IMPORTANT))
    You should handle the case where the order fails due to rate limiting (maybe wait and try again?)
    """

class Strategy:
    """Template for a strategy."""

    def __init__(self):
        """Your initialization code goes here."""

    self.times = 0
    self.holdings = {"A": 0, "B": 0, "C": 0}
    self.funds = 100000
    self.ABuy = {}
    self.ASell = {}
    self.ATrades = {}
    self.APrices = []
    self.BBuy = {}
    self.BSell = {}
    self.BTrades = {}
    self.BPrices = []
    self.CBuy = {}
    self.CSell = {}
    self.CTrades = {}
    self.CPrices = []

    self.A_Fair_Price = []
    self.B_Fair_Price = []
    self.C_Fair_Price = []

    def on_trade_update(self, ticker: str, side: str, price: float, quantity: float) -> None:
        """Called whenever two orders match. Could be one of your orders, or two other people's orders.

        Parameters
        ----------
        ticker
            Ticker of orders that were matched ("A", "B", or "C")
        side 
            Side of orders that were matched ("BUY" or "SELL")
        price
            Price that trade was executed at
        quantity
            Volume traded
        """
        print(f"Python Trade update: {ticker} {side} {price} {quantity}")

        if ticker == "A":
            if price in self.ATrades:
                self.ATrades[price] += quantity
            else:
                self.ATrades[price] = quantity
        if ticker == "B":
            if price in self.BTrades:
                self.BTrades[price] += quantity
            else:
                self.BTrades[price] = quantity

        else:
            if price in self.CTrades:
                self.CTrades[price] += quantity
            else:
                self.CTrades[price] = quantity

    def update_price(self, ticker):
        if ticker == "A":
            if self.ABuy == {} or self.ASell == {}:
                return
            self.APrices.append((min(list(self.ASell.keys())) + max(list(self.ABuy.keys())))/2)
            if len(self.APrices) > 50:
                self.APrices = self.APrices[-50:]
        elif ticker == "B":
            if self.BBuy == {} or self.BSell == {}:
                return
            self.BPrices.append((min(list(self.BSell.keys())) + max(list(self.BBuy.keys()))) / 2)
        else:
            if self.CBuy == {} or self.CSell == {}:
                return
            self.APrices.append((min(list(self.CSell.keys())) + max(list(self.CBuy.keys()))) / 2)


    def on_orderbook_update(
        self, ticker: str, side: str, price: float, quantity: float
    ) :
        """Called whenever the orderbook changes. This could be because of a trade, or because of a new order, or both.

        Parameters
        ----------
        ticker
            Ticker that has an orderbook update ("A", "B", or "C")
        side
            Which orderbook was updated ("BUY" or "SELL")
        price
            Price of orderbook that has an update
        quantity
            Volume placed into orderbook
        """
        self.times += 1
        self.update_data(ticker, side, price, quantity)
        self.update_VWAP(ticker)
        self.update_price(ticker)
        self.decision(ticker, self.funds // 3)
        self.A_Fair_Price = np.append(A_Fair_Price, calculate_fairprice_bsformula())
        print(f"Python Orderbook update: {ticker} {side} {price} {quantity}")


    def calculate_fairprice_bsformula(x, t_star, t, r, c, v):
        d1 = math.log(x / c) + (r + 1/2 * math.pow(v,2)  ) * (t_star - t)
        d2 = math.log(x / c) + (r - 1/2 * math.pow(v,2)  ) * (t_star - t)
        w = x * scipy.stats.norm.cdf(d1) - c * math.exp(r * (t - t_star)) * scipy.stats.norm.cdf(d2)
        return w


    def use_dqn(self):

        parser = argparse.ArgumentParser()
        train_options = parser.parse_args()
        check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR,TENSORBOARD_LOG_DIR, RESULTS_DIR])

        kwargs = {}

        if train_options.mode == "train":
            from finrl import train
            from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
            env = StockTradingEnv

            #plug in the parameters
            train(
                start_date=TRAIN_START_DATE,
                end_date=TRAIN_END_DATE,
                ticker_list={"A","B","C"},
                data_source="",
                time_interval = "1D",
                technical_indicator_list=INDICATORS,
                drl_lib="rllib",
                env=env,
                model_name="dqn",
                cwd="./test_dqn",
                rllib_params=RLlib_PARAMS,
                total_episodes=30,

            )

        """
        if train_options.mode == "test":
            from finrl import test
        from finrl import test
        env = StockTradingEnv

        kwargs = {}
        # plug in the parameters
        test()

        """

        else if train_options.mode == "trade":
            from finrl import trade
            env = StockTradingEnv

        kwargs = {}
        #trade with bid/ask decisions and update data
        trade(
            start_date=TRADE_START_DATE,
            end_date=TRADE_END_DATE,
            ticker_list={"A","B","C"},
            data_source="",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="rllib",
            env=env,
            model_name="dqn",
            if_vix=True,
            kwargs=kwargs,)

    def find_optimal_strategy(X, A, B, C, Q, R, x):

        #computing the Riccati equation at time T for asset i
        P_Dot = P @ B @ np.linalg.inv(R) @B.T @ P - A.T @ P - P @ A - Q
        q_Dot = np.trace(C.T @ P @ C)

        #solve the Riccati equation with control.lib
        X,L,G = control.dare*(A,B,Q,R)

        #compute optimal value of trade
        int_q = X + np.trace(C.T @ X @ C)
        V = x.T @ X @ x + int_q
        U = (-1) * np.linalg.inv(R) @ B.T @ P * x

       return V,U


     def decision(self,ticker, Fair_Price):
         #compute a fair price for three tickers
         from statsmodels.tsa.api import VAR
         self.A_Fair_Price = calculate_fairprice_bsformula(self.APrices,trade_time,self.times,r,c,VAR(self.APrices))
         self.B_Fair_Price = calculate_fairprice_bsformula(self.BPrices,trade_time,self.times,r,c,VAR(self.BPrices))
         self.C_Fair_Price = calculate_fairprice_bsformula(self.CPrices,trade_time,self.times,r,c,VAR(self.CPrices))

         #predict exercise price through dqn





    def on_trade_update(self, ticker: str, side, price, quantity):
        """Called whenever two orders match. Could be one of your orders, or two other people's orders.

        Parameters
        ----------
        ticker
            Ticker of orders that were matched ("A", "B", or "C")
        side
            Side of orders that were matched ("BUY" or "SELL")
        price
            Price that trade was executed at
        quantity
            Volume traded
        """
        self.times += 1
        self.update_price(ticker)
        self.update_trades(ticker, price, quantity)
        self.decision(ticker, self.funds // 3)

    def on_orderbook_update(self, ticker, side, price, quantity):
        """Called whenever the orderbook changes. This could be because of a trade, or because of a new order, or both.

        Parameters
        ----------
        ticker
            Ticker that has an orderbook update ("A", "B", or "C")
        side
            Which orderbook was updated ("BUY" or "SELL")
        price
            Price of orderbook that has an update
        quantity
            Volume placed into orderbook
        """
        self.times += 1
        self.update_data(ticker, side, price, quantity)
        self.update_VWAP(ticker)
        self.update_price(ticker)
        self.decision(ticker, self.funds // 3)

    def on_account_update(
        self,
        ticker: str,
        side: str,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) :
        """Called whenever one of your orders is filled.

        Parameters
        ----------
        ticker
            Ticker of order that was fulfilled ("A", "B", or "C")
        side
            Side of order that was fulfilled ("BUY" or "SELL")
        price
            Price that order was fulfilled at
        quantity
            Volume of order that was fulfilled
        capital_remaining
            Ammount of capital after fulfilling order
        """
        print(
            f"Python Account update: {ticker} {side} {price} {quantity} {capital_remaining}"
        )