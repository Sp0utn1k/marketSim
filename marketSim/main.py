from utils import load_data
from marketSim.instruments.instruments import instruments, ref_currencies
from collections import OrderedDict
import numpy as np
import time
import random
import copy


class Wallet:
    def __init__(self, start_currency="USD", start_val=1,
                 available_markets=None, fees=7.5e-4, episode_len=200,
                 pred_len=300, interval=60):

        if available_markets is None:
            available_markets = ["BTCUSD"]
        self.i = None
        self.episode0 = None
        self.links = None
        self.data_size = None
        self.pred_len = pred_len
        self.episode_len = episode_len
        self.start_currency = start_currency
        self.start_val = float(start_val)
        self.interval = interval
        self.action_space = (len(available_markets))
        self.fees = fees

        if available_markets == 'all':
            available_markets = instruments
        self.market_data = OrderedDict()
        self.wallet = OrderedDict()  # Ensure values are always given in the same order
        self.t0 = 0
        self.tend = time.time()
        self.init_markets(available_markets)
        self.init_links()

    def init_markets(self, available_markets):
        for mrkt in available_markets:
            quote, base = market_to_currencies(mrkt)
            self.wallet[quote] = 0.0
            self.wallet[base] = 0.0

            market_data = load_data(mrkt, to_torch=False)
            t0 = market_data[0, 0]
            tend = market_data[-1, 0]
            self.t0 = max(t0, self.t0)
            self.tend = min(tend, self.tend)
            self.market_data[mrkt] = market_data

        self.data_size = (self.tend - self.t0) / self.interval + 1
        self.data_size = int(self.data_size)

        for (mrkt, market_data) in self.market_data.items():
            market_data = market_data[market_data[:, 0] <= self.tend, :]
            market_data = fill_empty(market_data, self.tend)
            market_data = market_data[market_data[:, 0] >= self.t0, :]
            self.market_data[mrkt] = market_data

    def init_links(self):
        self.links = {self.start_currency: []}
        i = 0
        while len(self.links.keys()) < len(self.wallet.keys()):
            for mrkt in self.market_data.keys():
                quote, base = market_to_currencies(mrkt)
                for curr in list(self.links.keys()):
                    if len(self.links[curr]) == i:
                        if quote == curr:
                            other = base
                        elif base == curr:
                            other = quote
                        else:
                            continue
                        if other not in self.links.keys():
                            self.links[other] = self.links[curr] + [mrkt]
            i += 1

    def get_wallet_values(self):
        return np.array([val for val in self.wallet.values()])

    def get_wallet(self):
        return [(key, val) for (key, val) in self.wallet.items()]

    def get_market_data_values(self):

        # returns an array of form (sequence,markets,(O,H,L,C,V))
        i2 = self.episode0 + self.i
        i1 = i2 - self.pred_len
        market_data = np.stack([market_val[i1:i2, 1:] for market_val in self.market_data.values()])
        t = self.t0 + i1 * self.interval
        return t, market_data.transpose(1, 0, 2)

    def reset(self):

        self.i = 0
        self.episode0 = random.randrange(self.pred_len, self.data_size - self.episode_len)
        for marketname in self.market_data.keys():
            quote, base = market_to_currencies(marketname)
            self.wallet[quote] = 0.0
            self.wallet[base] = 0.0
        self.wallet[self.start_currency] = self.start_val
        return self.get_state()

    def get_state(self):

        t, market_data = self.get_market_data_values()
        wallet = np.append(self.get_wallet_values(), t)
        return wallet, market_data

    def transaction(self, marketname, qtt, apply_fees=True):
        quote, base = market_to_currencies(marketname)
        rate = self.market_data[marketname][self.episode0 + self.i - 1, 4]

        if qtt == 0:
            return

        if qtt > 0:  # buy
            d_base = -qtt
            d_quote = qtt / rate * (1 - self.fees * apply_fees)
        else:  # sell
            qtt *= -1
            d_quote = -qtt
            d_base = qtt * rate * (1 - self.fees * apply_fees)

        self.wallet[quote] += d_quote
        self.wallet[base] += d_base

    def get_reward(self):

        equivalents = copy.deepcopy(self.wallet)
        links_cp = copy.deepcopy(self.links)
        for curr, val in equivalents.items():
            transition_curr = curr
            while len(links_cp[curr]):
                mrkt = links_cp[curr].pop()
                quote, base = market_to_currencies(mrkt)
                rate = self.market_data[mrkt][self.i + self.episode0, 4]
                if transition_curr == base:
                    transition_curr = quote
                    rate = 1 / rate
                else:
                    transition_curr = base
                equivalents[curr] *= rate

        return sum(equivalents.values())

    def apply_actions(self, actions):

        transactions_per_currency = {}
        for currency in self.wallet.keys():
            transactions_per_currency[currency] = []

        for i, marketname in enumerate(self.market_data.keys()):
            quote, base = market_to_currencies(marketname)
            if actions[i] > 0:
                transactions_per_currency[base] += [[marketname, actions[i]]]
            else:
                transactions_per_currency[quote] += [[marketname, actions[i]]]

        for currency, transactions in transactions_per_currency.items():
            proportions = [abs(transaction[1]) for transaction in transactions]
            divider = max(sum(proportions), 1)
            multiplier = self.wallet[currency] / divider

            for i in range(len(transactions)):
                transactions_per_currency[currency][i][1] *= multiplier

        print(transactions_per_currency)
        for transactions in transactions_per_currency.values():
            for transaction in transactions:
                marketname = transaction[0]
                action = transaction[1]
                self.transaction(marketname, action)

    def step(self, actions):

        self.apply_actions(actions)

        reward = self.get_reward()

        self.i += 1
        if self.i == self.episode_len:
            is_done = True
            state = None
        else:
            is_done = False
            state = self.get_state()

        return state, reward, is_done

    def print_wallet(self):
        for currency, value in self.wallet.items():
            print(currency, value)


def market_to_currencies(market_name):
    for rc in ref_currencies:
        if rc == market_name[-len(rc):]:
            base = rc
            quote = market_name[:-len(rc)].split('.')[0]
            if base == 'XBT':
                base = 'BTC'
            if quote == 'XBT':
                quote = 'BTC'
            return quote, base


def fill_empty(old_data, tend, interval=60.0):
    t0 = old_data[0, 0]
    size = (tend - t0) / interval + 1

    if old_data[-1, 0] < tend:
        val = old_data[-1, 4]
        sample = np.array([tend, val, val, val, val, 0]).reshape(1, -1)
        old_data = np.concatenate((old_data, sample))

    new_data = np.zeros((int(size), 6))
    new_data[0, :] = old_data[0, :]
    i = 1
    i_new = 1
    val = new_data[0, 4]
    while i < len(old_data):

        t1 = old_data[i - 1, 0]
        t2 = old_data[i, 0]
        di = (t2 - t1) / interval
        N = int(di) - 1
        if di > 1:
            t = t1 + interval * np.arange(1, N + 1).reshape(N, 1)
            candles = val * np.ones((N, 4))
            volumes = np.zeros((N, 1))
            new_data[i_new:i_new + N, :] = np.concatenate((t, candles, volumes), axis=1)

        new_data[i_new + N, :] = old_data[i, :]

        i_new += int(di)
        val = old_data[i, 4]
        i += 1

    return new_data


if __name__ == '__main__':

    env = Wallet(start_currency="EUR", start_val=1, episode_len=40,
                 available_markets=["BTCUSD", "ETHXBT", "ETHUSD", "EURUSD", "ETHUSDT"])

    for market, data in env.market_data.items():
        print(market, data.shape)

    print("")
    env.print_wallet()

    s = env.reset()

    done = False
    while not done:
        a = [random.uniform(-1, 1) for _ in range(env.action_space)]
        s, r, done = env.step(a)

        print("")
        print(a)
        env.print_wallet()
        print(r)
