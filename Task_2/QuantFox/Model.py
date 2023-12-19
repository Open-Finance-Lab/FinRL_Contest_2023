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


class Strategy:
    MAX_LEN = 100
    time_frame = 15
    slope_ma_period = 20
    angle_ma_mode = 0
    angle_ma_price = 0
    k_count = 5

    def __init__(self):
        self.times = 0
        self.holdings = {"A": 0, "B": 0, "C": 0}
        self.funds = 100000

        # Use Pandas dataframes to store orderbook and trade data
        self.orderbooks = {"A": pd.DataFrame(columns=["price", "quantity", 'side']),
                           "B": pd.DataFrame(columns=["price", "quantity", 'side']),
                           "C": pd.DataFrame(columns=["price", "quantity", 'side'])}

        self.trades = {"A": pd.DataFrame(columns=["price", "quantity", 'side']),
                       "B": pd.DataFrame(columns=["price", "quantity", 'side']),
                       "C": pd.DataFrame(columns=["price", "quantity", 'side'])}

        self.vwaps = {"A": pd.DataFrame(columns=["value"]),
                      "B": pd.DataFrame(columns=["value"]),
                      "C": pd.DataFrame(columns=["value"])}

        self.prices = {"A": pd.DataFrame(columns=["value", 'RSI', 'CCI', 'Slope']),
                       "B": pd.DataFrame(columns=["value", 'RSI', 'CCI', 'Slope']),
                       "C": pd.DataFrame(columns=["value", 'RSI', 'CCI', 'Slope'])}

    def rma(self, x, n, y0):
        a = (n - 1) / n
        ak = a ** np.arange(len(x) - 1, -1, -1)
        return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a ** np.arange(1, len(x) + 1)]

    def update_rsi(self, ticker, n=14):
        df = pd.DataFrame(self.prices[ticker]["value"])
        if df.shape[0] < n+1:
            return
        df['change'] = df['value'].diff()
        df['gain'] = df.change.mask(df.change < 0, 0.0)
        df['loss'] = -df.change.mask(df.change > 0, -0.0)
        df['avg_gain'] = self.rma(df.gain[n + 1:].to_numpy(), n, np.nansum(df.gain.to_numpy()[:n + 1]) / n)
        df['avg_loss'] = self.rma(df.loss[n + 1:].to_numpy(), n, np.nansum(df.loss.to_numpy()[:n + 1]) / n)
        df['rs'] = df.avg_gain / df.avg_loss
        df['rsi_14'] = 100 - (100 / (1 + df.rs))
        self.prices[ticker]["RSI"] = df['rsi_14']

    def update_cci(self, ticker, ndays=24):
        df = pd.DataFrame(self.prices[ticker]["value"])
        if df.shape[0] < ndays:
            return
        df['TP'] = df['value']
        df['sma'] = df['TP'].rolling(ndays).mean()
        df['mad'] = df['TP'].rolling(ndays).apply(lambda x: pd.Series(x).mad())
        df['CCI'] = (df['TP'] - df['sma']) / (0.015 * df['mad'])
        self.prices[ticker]['CCI'] = df['CCI']

    def calculate_slope_indicator(self, ticker):
        if self.prices[ticker].shape[0] < max(self.slope_ma_period, self.time_frame):
            return
        # Calculate moving averages
        ma_1 = self.prices[ticker]['value'].rolling(window=self.slope_ma_period).mean()
        ma_2 = self.prices[ticker]['value'].rolling(window=self.slope_ma_period).mean().shift(self.k_count)

        # Calculate Angle
        self.prices[ticker]['Slope'] = 0  # Initialize Angle column with zeros
        self.prices[ticker]['Slope'][self.time_frame:] = np.arctan((ma_1 - ma_2) / (self.time_frame * 8)) * 180 / np.pi

    def update_orderbook_data(self, ticker, side, price, quantity):
        df: pd.DataFrame = self.orderbooks[ticker]
        price_int = int(price * 100)
        pos_idx = df.loc[(df['price'] == price_int) & (df['side'] == side)]
        if pos_idx.shape[0] == 0:
            new_row = pd.DataFrame({"price": [price_int], "quantity": [quantity], "side": [side]})
            self.orderbooks[ticker] = pd.concat([df, new_row], ignore_index=True)
        else:
            df.loc[(df['price'] == price_int) & (df['side'] == side), 'quantity'] += quantity

        positions_to_drop = self.orderbooks[ticker].index[df['price'] == 0].tolist()

        if positions_to_drop:
            # Drop rows with value = 0
            self.orderbooks[ticker] = self.orderbooks[ticker].drop(positions_to_drop)

        self.orderbooks[ticker] = self.orderbooks[ticker].tail(self.MAX_LEN).reset_index(drop=True)

    # def update_VWAP(self, ticker):
    #     df = pd.DataFrame(self.trades[ticker])
    #     MAX_LEN_VWAP = 50
    #     # Calculate the cumulative sums within the window
    #     df['cumulative_price_volume'] = df['price'] * df['quantity']
    #     df['cumulative_price_volume'] = df['cumulative_price_volume'].rolling(window=MAX_LEN_VWAP).sum()
    #     df['cumulative_volume'] = df['quantity'].rolling(window=MAX_LEN_VWAP).sum()
    #
    #     # Calculate the VWAP
    #     df['value'] = df['cumulative_price_volume'] / df['cumulative_volume']
    #
    #     self.vwaps[ticker] = df[['value']].copy().tail(MAX_LEN_VWAP).reset_index(drop=True)

    def update_price_by_order(self, ticker):
        df = self.orderbooks[ticker]
        # Find the max price when side = 'BUY'
        max_buy_price = df.loc[df['side'] == 'BUY', 'price'].max()

        # Find the min price when side = 'SELL'
        min_sell_price = df.loc[df['side'] == 'SELL', 'price'].min()

        if np.isnan(min_sell_price) or np.isnan(max_buy_price) or min_sell_price < max_buy_price:
            return

        new_row = pd.DataFrame({"value": [int((min_sell_price + max_buy_price) / 2 * 100)]})
        df = pd.concat([self.prices[ticker], new_row], ignore_index=True)
        self.prices[ticker] = df.tail(self.MAX_LEN).reset_index(drop=True)
        self.update_rsi(ticker)
        # self.update_cci(ticker)
        self.calculate_slope_indicator(ticker)

    def update_price_by_trade(self, ticker, price, quantity, side):
        price_int = int(price * 100)
        df = pd.concat([self.prices[ticker], pd.DataFrame({'value': [price_int]})], ignore_index=True)
        self.prices[ticker] = df.tail(self.MAX_LEN).reset_index(drop=True)
        self.update_rsi(ticker)
        # self.update_cci(ticker)
        self.calculate_slope_indicator(ticker)

    def update_trades(self, ticker, price, quantity, side):
        df: pd.DataFrame = self.trades[ticker]
        price_int = int(price * 100)
        pos_idx = df.loc[(df['price'] == price_int) & (df['side'] == side)]
        if pos_idx.shape[0] == 0:
            new_row = pd.DataFrame({"price": [price_int], "quantity": [quantity], "side": [side]})
            self.trades[ticker] = pd.concat([df, new_row], ignore_index=True)
        else:
            df.loc[(df['price'] == price_int) & (df['side'] == side), 'quantity'] += quantity
        self.trades[ticker] = self.trades[ticker].tail(self.MAX_LEN).reset_index(drop=True)

    def decision(self, ticker, funding):
        # if len(self.vwaps[ticker]['value'].tail(1).values) == 0:
        #     return
        # last_vwap_value = self.vwaps[ticker]['value'].tail(1).values[0]
        # ob = self.orderbooks[ticker]
        # sell_quantity_sum = ob.loc[ob['side'] == 'SELL', 'quantity'].sum()
        # min_sell_price = ob.loc[ob['side'] == 'SELL', 'price'].min() / 100.0
        # if sell_quantity_sum > 0 and self.holdings["A"] == 0:
        #     place_market_order("BUY", ticker, limit // min_sell_price, min_sell_price)
        # if last_vwap_value == np.NAN or last_vwap_value == 0 or self.prices[ticker].shape[0] < 2:
        #     return
        #
        # price_tails = self.prices[ticker]['value'].tail(2).values
        # if last_vwap_value * 100 < price_tails[0] < price_tails[1]:
        #     place_market_order("SELL", ticker, self.holdings[ticker], price_tails[0] / 100.0)
        # elif last_vwap_value * 100 > price_tails[0] > price_tails[1]:
        #     place_market_order("BUY", ticker, limit // (price_tails[1] / 100.0), price_tails[1] / 100.0)
        # else:
        #     return
        if len(self.prices[ticker]['value'].tail(1).values) > 0:
            price = self.prices[ticker]['value'].tail(1).values[0]
        else:
            return

        if len(self.prices[ticker]['RSI'].tail(1).values) > 0:
            rsi = self.prices[ticker]['RSI'].tail(1).values[0]
        else:
            rsi = 0
        # cci = self.prices[ticker]['CCI'].tail(1).values[0]
        if len(self.prices[ticker]['Slope'].tail(1).values) > 0:
            slope = self.prices[ticker]['Slope'].tail(1).values[0]
        else:
            slope = 0

        if -0.5 < slope < 0.5:
            return

        if rsi > 70 and funding // (price / 100.0) > 0:
            place_market_order("BUY", ticker, funding // (price / 100.0), price / 100.0)
        elif rsi < 30 and self.holdings[ticker] > 0:
            place_market_order("SELL", ticker, self.holdings[ticker], price / 100.0)

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
        self.times += 1
        self.update_price_by_trade(ticker, price, quantity, side)
        self.update_trades(ticker, price, quantity, side)
        self.decision(ticker, self.funds // 3)

    def on_orderbook_update(
            self, ticker: str, side: str, price: float, quantity: float
    ) -> None:
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
        self.update_orderbook_data(ticker, side, price, quantity)
        # self.update_VWAP(ticker)
        self.update_price_by_order(ticker)
        self.decision(ticker, self.funds // 3)

    def on_account_update(
            self,
            ticker: str,
            side: str,
            price: float,
            quantity: float,
            capital_remaining: float,
    ) -> None:
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
        self.funds = capital_remaining
        if side == "BUY":
            self.holdings[ticker] += quantity
        else:
            self.holdings[ticker] -= quantity


if __name__ == '__main__':
    strategy = Strategy()
    strategy.on_orderbook_update("A", "BUY", 20.05, 100)