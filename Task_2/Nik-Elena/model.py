import numpy as np
import pandas as pd

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
    
    print(side, ticker, quantity, price)

class Strategy:

    def __init__(self) -> None:
        self.order_history = []
        self.position = {}
        self.initial_capital = 100000
        self.current_capital = self.initial_capital
        self.price_history = {'A': [], 'B': [], 'C': []}
        self.volume_history = {'A': [], 'B': [], 'C': []}
        self.buy_volume = {'A': 0, 'B': 0, 'C': 0}
        self.sell_volume = {'A': 0, 'B': 0, 'C': 0}
        self.order_book_imbalance = { 'A': 0, 'B': 0, 'C': 0 }
        self.mean_window = 30  # Num periods to calculate mean
        self.std_dev_multiplier = 2  # Standard deviation multiplier in Bollinger Bands
        self.imbalance_threshold = 0.6  # Order book imbalance threshold
        self.vwap = {'A': [], 'B': [], 'C': []}

    def on_trade_update(self, ticker: str, side: str, price: float, quantity: float) -> None:
        self.price_history[ticker].append(price)
        self.volume_history[ticker].append(quantity)
        
        if side == 'BUY':
            self.buy_volume[ticker] += quantity
        else:
            self.sell_volume[ticker] += quantity

        self.calculate_vwap(ticker)
        self.check_trading_opportunity(ticker)

    def on_orderbook_update(self, ticker: str, side: str, price: float, quantity: float) -> None:
        if side == 'BUY':
            self.order_book_imbalance[ticker] += quantity
        else:
            self.order_book_imbalance[ticker] -= quantity

        total_volume = sum([abs(q) for q in self.order_book_imbalance.values()])
        if total_volume > 0:
            imbalance_ratio = self.order_book_imbalance[ticker] / total_volume
            if abs(imbalance_ratio) > self.imbalance_threshold:
                self.execute_order_book_strategy(ticker, imbalance_ratio, price)

    def on_account_update(self, ticker: str, side: str, price: float, quantity: float, capital_remaining: float) -> None:
        self.current_capital = capital_remaining

    def check_trading_opportunity(self, ticker: str):
        if len(self.price_history[ticker]) >= self.mean_window:
            prices = np.array(self.price_history[ticker][-self.mean_window:])
            moving_avg = np.mean(prices)
            std_dev = np.std(prices)
            upper_band = moving_avg + (self.std_dev_multiplier * std_dev)
            lower_band = moving_avg - (self.std_dev_multiplier * std_dev)
            current_price = self.price_history[ticker][-1]
            z_score = (current_price - moving_avg) / std_dev if std_dev != 0 else 0

            vwap = self.vwap.get(ticker)

            if current_price <= lower_band and z_score < -2 and current_price < vwap and self.current_capital >= current_price:
                # The price is below the lower Bollinger Band, Z-score is significantly low, and below VWAP -> buy signal
                quantity = self.current_capital // current_price
                self.place_market_order('BUY', ticker, quantity, current_price)
            elif current_price >= upper_band and z_score > 2 and current_price > vwap and self.position.get(ticker, 0) > 0:
                # The price is above the upper Bollinger Band, Z-score is significantly high, and above VWAP -> sell signal
                self.place_market_order('SELL', ticker, self.position[ticker], current_price)


    def execute_order_book_strategy(self, ticker: str, imbalance_ratio: float, price: float):
        if imbalance_ratio > 0 and self.current_capital >= price:
            quantity = self.current_capital // price
            self.place_market_order('BUY', ticker, quantity, price)
        elif imbalance_ratio < 0 and self.position.get(ticker, 0) > 0:
            self.place_market_order('SELL', ticker, self.position[ticker], price)

    def place_market_order(self, side: str, ticker: str, quantity: float, price: float) -> bool:
        success = place_market_order(side, ticker, quantity, price)
        if success:
            self.order_history.append((ticker, side, price, quantity))
            if side == 'BUY':
                self.position[ticker] = self.position.get(ticker, 0) + quantity
                self.current_capital -= price * quantity
            else:
                self.position[ticker] = self.position.get(ticker, 0) - quantity
                self.current_capital += price * quantity
        return success
    
    def calculate_vwap(self, ticker: str):
        if not self.price_history[ticker]:
            return
        prices = self.price_history[ticker]
        volumes = self.volume_history[ticker]
        df = pd.DataFrame({'price': prices, 'volume': volumes})
        df['cumulative_quantity'] = df['volume'].cumsum()
        df['cumulative_turnover'] = (df['price'] * df['volume']).cumsum()
        vwap_value = df['cumulative_turnover'].iloc[-1] / df['cumulative_quantity'].iloc[-1]
        self.vwap[ticker] = vwap_value