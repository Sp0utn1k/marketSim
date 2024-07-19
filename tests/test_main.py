# tests/test_main.py

import pytest
from src.main import Wallet, market_to_currencies, fill_empty
import numpy as np


@pytest.fixture
def wallet():
    return Wallet(start_currency="EUR", start_val=1000, episode_len=40,
                  available_markets=["BTCUSD", "ETHXBT", "ETHUSD", "EURUSD", "ETHUSDT"])


def test_wallet_initialization(wallet):
    assert wallet.start_currency == "EUR"
    assert wallet.start_val == 1000
    assert wallet.episode_len == 40
    assert wallet.action_space == 5  # 5 markets


def test_wallet_reset(wallet):
    state = wallet.reset()
    assert len(state) == 2
    assert isinstance(state[0], np.ndarray)
    assert isinstance(state[1], np.ndarray)


def test_transaction_buy(wallet):
    wallet.reset()
    initial_eur = wallet.wallet["EUR"]
    initial_btc = wallet.wallet["BTC"]

    wallet.make_transaction("BTCEUR", 100)  # Buy BTC
    assert wallet.wallet["EUR"] < initial_eur
    assert wallet.wallet["BTC"] > initial_btc


def test_transaction_sell(wallet):
    wallet.reset()
    wallet.wallet["BTC"] = 1  # Give some BTC for selling
    initial_eur = wallet.wallet["EUR"]
    initial_btc = wallet.wallet["BTC"]

    wallet.make_transaction("BTCEUR", -0.5)  # Sell BTC
    assert wallet.wallet["EUR"] > initial_eur
    assert wallet.wallet["BTC"] < initial_btc


def test_market_to_currencies():
    assert market_to_currencies("BTCUSD") == ("BTC", "USD")
    assert market_to_currencies("ETHXBT") == ("ETH", "BTC")


def test_fill_empty():
    old_data = np.array([[0, 1, 1, 1, 1, 0], [60, 2, 2, 2, 2, 0]])
    tend = 180
    filled_data = fill_empty(old_data, tend)
    assert filled_data.shape == (4, 6)
    assert filled_data[1, 0] == 60
    assert filled_data[-1, 0] == 180
    assert filled_data[-1, 1] == 2  # Last known value

    old_data = np.array([[0, 1, 1, 1, 1, 0], [120, 2, 2, 2, 2, 0]])
    tend = 240
    filled_data = fill_empty(old_data, tend)
    assert filled_data.shape == (5, 6)
    assert filled_data[1, 0] == 60
    assert filled_data[2, 0] == 120
    assert filled_data[-1, 0] == 240
    assert filled_data[-1, 1] == 2  # Last known value


if __name__ == '__main__':
    pytest.main()
