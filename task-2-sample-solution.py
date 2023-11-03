#description: Estimate market entry points using VWAP and derivative trends.
def place_market_order(side, ticker, quantity, price):
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

    def __init__(self):
        self.times = 0
        self.holdings = {"A": 0, "B": 0, "C": 0}
        self.funds = 100000
        self.ABuy = {}
        self.ASell = {}
        self.ATrades = {}
        self.AVWAP = []
        self.APrices = []
        self.BBuy = {}
        self.BSell = {}
        self.BTrades = {}
        self.BVWAP = []
        self.BPrices = []
        self.CBuy = {}
        self.CSell = {}
        self.CTrades = {}
        self.CVWAP = []
        self.CPrices = []

    def update_data(self, ticker, side, price, quantity):
        if side == "BUY":
            if ticker == "A":
                if price in self.ABuy:
                    self.ABuy[price] += quantity
                else:
                    self.ABuy[price] = quantity
            elif ticker == "B":
                if price in self.BBuy:
                    self.BBuy[price] += quantity
                else:
                    self.BBuy[price] = quantity
            else:
                if price in self.CBuy:
                    self.CBuy[price] += quantity
                else:
                    self.CBuy[price] = quantity
        else:
            if ticker == "A":
                if price in self.ASell:
                    self.ASell[price] += quantity
                else:
                    self.ASell[price] = quantity
            elif ticker == "B":
                if price in self.BSell:
                    self.BSell[price] += quantity
                else:
                    self.BSell[price] = quantity
            else:
                if price in self.CSell:
                    self.CSell[price] += quantity
                else:
                    self.CSell[price] = quantity

    def update_VWAP(self, ticker):
        cumulative_volume = 0
        cumulative_price_volume = 0

        if ticker == "A":
            if self.ATrades == {}:
                return
            for price, volume in zip(list(self.ATrades.keys()), list(self.ATrades.values())):
                cumulative_volume += volume
                cumulative_price_volume += price * volume

            self.AVWAP.append(cumulative_price_volume / cumulative_volume)
            if len(self.AVWAP) > 50:
                self.AVWAP = self.AVWAP[-50:]

        elif ticker == "B":
            if self.BTrades == {}:
                return
            for price, volume in zip(list(self.BTrades.keys()), list(self.BTrades.values())):
                cumulative_volume += volume
                cumulative_price_volume += price * volume

            self.BVWAP.append(cumulative_price_volume / cumulative_volume)
            if len(self.BVWAP) > 50:
                self.BVWAP = self.BVWAP[-50:]

        else:
            if self.CTrades == {}:
                return
            for price, volume in zip(list(self.CTrades.keys()), list(self.CTrades.values())):
                cumulative_volume += volume
                cumulative_price_volume += price * volume

            self.CVWAP.append(cumulative_price_volume / cumulative_volume)
            if len(self.CVWAP) > 50:
                self.CVWAP = self.CVWAP[-50:]

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

    def update_trades(self, ticker, price, quantity):
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

    def decision(self, ticker, limit):
        if ticker == "A":
            if self.ASell != {} and self.holdings["A"] == 0:
                place_market_order("BUY", "A", limit//min(list(self.ASell.keys())), min(list(self.ASell.keys())))
            if len(self.AVWAP) == 0 or len(self.APrices) < 2:
                return
            if self.AVWAP[-1] < self.APrices[-2] < self.APrices[-1]:
                place_market_order("SELL", "A", self.holdings["A"], self.APrices[-2])
            elif self.AVWAP[-1] > self.APrices[-2] > self.APrices[-1]:
                place_market_order("BUY", "A", limit // self.APrices[-1], self.APrices[-1])
            else:
                return

        elif ticker == "B":
            if self.BSell != {} and self.holdings["B"] == 0:
                place_market_order("BUY", "B", limit // min(list(self.BSell.keys())), min(list(self.BSell.keys())))
            if len(self.BVWAP) == 0 or len(self.BPrices) < 2:
                return
            if self.BVWAP[-1] < self.BPrices[-2] < self.BPrices[-1]:
                place_market_order("SELL", "B", self.holdings["B"], self.BPrices[-2])
            elif self.BVWAP[-1] > self.BPrices[-2] > self.BPrices[-1]:
                place_market_order("BUY", "B", limit // self.BPrices[-1], self.BPrices[-1])
            else:
                return

        else:
            if self.CSell != {} and self.holdings["C"] == 0:
                place_market_order("BUY", "C", limit // min(list(self.CSell.keys())), min(list(self.CSell.keys())))
            if len(self.CVWAP) == 0 or len(self.CPrices) < 2:
                return
            if self.CSell != {} and self.holdings["C"] == 0:
                place_market_order("BUY", "C", min(list(self.CSell.keys())))
                return
            if self.CVWAP[-1] < self.CPrices[-2] < self.CPrices[-1]:
                place_market_order("SELL", "C", self.holdings["C"], self.CPrices[-2])
            elif self.CVWAP[-1] > self.CPrices[-2] > self.CPrices[-1]:
                place_market_order("BUY", "C", limit // self.CPrices[-1], self.CPrices[-1])
            else:
                return

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

    def on_account_update(self, ticker, side, price, quantity, capital_remaining):
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
        self.funds = capital_remaining - 10000
        if side == "BUY":
            if ticker == "A":
                self.holdings["A"] += quantity
            elif ticker == "B":
                self.holdings["B"] += quantity
            else:
                self.holdings["C"] += quantity
        else:
            if ticker == "A":
                self.holdings["A"] -= quantity
            elif ticker == "B":
                self.holdings["B"] -= quantity
            else:
                self.holdings["C"] -= quantity










