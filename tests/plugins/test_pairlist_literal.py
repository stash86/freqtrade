from unittest.mock import MagicMock

import pytest

from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.plugins.pairlist.RemotePairList import RemotePairList
from freqtrade.plugins.pairlistmanager import PairListManager
from tests.conftest import get_patched_exchange


def test_expand_pairlist_literal_order_and_duplicates():
    pairs = ["eth/usdt", "btc/usdt", "ETH/USDT", "missing/usdt"]
    markets = ["BTC/USDT", "eth/USDT", "ETH/usdt", "BTC/USDT"]

    assert expand_pairlist(pairs, markets, use_regex=False) == [
        "eth/USDT",
        "ETH/usdt",
        "BTC/USDT",
        "BTC/USDT",
        "eth/USDT",
        "ETH/usdt",
    ]
    assert pairs == ["eth/usdt", "btc/usdt", "ETH/USDT", "missing/usdt"]
    assert markets == ["BTC/USDT", "eth/USDT", "ETH/usdt", "BTC/USDT"]


def test_expand_pairlist_literal_metacharacters():
    pairs = ["A.B/USDT", "*UP/USDT", "[ABC]/USDT", ".*/USDT", "NO[PE/USDT"]
    markets = [
        "AXB/USDT",
        "A.B/USDT",
        "BTCUP/USDT",
        "*UP/USDT",
        "A/USDT",
        "[ABC]/USDT",
        "NO[PE/USDT",
    ]

    assert expand_pairlist(pairs, markets, use_regex=False) == [
        "A.B/USDT",
        "*UP/USDT",
        "[ABC]/USDT",
        "NO[PE/USDT",
    ]


@pytest.mark.parametrize(
    "keep_invalid,expected",
    [
        (False, ["ETH/USDT", "BB_BTC/USDT", "A.B/USDT"]),
        (True, ["ETH/USDT", "missing/USDT:USDT", "BTC/USD"]),
    ],
)
def test_expand_pairlist_literal_keep_invalid(keep_invalid, expected):
    assert (
        expand_pairlist(
            [
                "eth/usdt",
                "missing/USDT:USDT",
                "BB_BTC/USDT",
                "A.B/USDT",
                ".*/USDT",
                "BTC/USD",
                "*UP/USDT",
            ],
            ["ETH/USDT", "BB_BTC/USDT", "A.B/USDT", "BTC/USDT"],
            keep_invalid=keep_invalid,
            use_regex=False,
        )
        == expected
    )


@pytest.mark.parametrize(
    "pair,markets,expected",
    [
        ("s/USDT", ["S/USDT", "\u017f/USDT"], ["S/USDT", "\u017f/USDT"]),
        (
            "s/USDT",
            [
                "\u017f/USDT",
                "K/USDT",
                "S/USDT",
                "\u017f/USDT",
                "s/USDT",
                "S/USDT",
                "\u212a/USDT",
            ],
            ["\u017f/USDT", "S/USDT", "\u017f/USDT", "s/USDT", "S/USDT"],
        ),
        ("\u017f/USDT", ["S/USDT"], ["S/USDT"]),
        ("k/USDT", ["K/USDT", "\u212a/USDT"], ["K/USDT", "\u212a/USDT"]),
        ("\u212a/USDT", ["K/USDT"], ["K/USDT"]),
        (
            "i/USDT",
            ["I/USDT", "\u0130/USDT", "\u0131/USDT"],
            ["I/USDT", "\u0130/USDT", "\u0131/USDT"],
        ),
        ("\u0130/USDT", ["I/USDT"], ["I/USDT"]),
        ("\u0131/USDT", ["I/USDT"], ["I/USDT"]),
        ("\u00df/USDT", ["SS/USDT", "\u1e9e/USDT"], ["\u1e9e/USDT"]),
        ("ss/USDT", ["\u00df/USDT", "SS/USDT"], ["SS/USDT"]),
    ],
)
def test_expand_pairlist_literal_unicode_case_matching(pair, markets, expected):
    # Literal strings retain the case equivalences of the existing regex mode.
    assert expand_pairlist([pair], markets, use_regex=False) == expected
    assert expand_pairlist([pair], markets) == expected


@pytest.mark.parametrize(
    "use_regex,expected",
    [
        (None, ["ETH/USDT", "XRP/USDT"]),
        (True, ["ETH/USDT", "XRP/USDT"]),
        (False, ["ETH/USDT"]),
    ],
)
def test_remote_pairlist_literal_config_and_cache(
    mocker, default_conf, markets, use_regex, expected
):
    default_conf["stake_currency"] = "USDT"
    default_conf["exchange"]["pair_blacklist"] = []
    pairlist_config = {
        "method": "RemotePairList",
        "pairlist_url": "https://example.com/pairlist",
        "refresh_period": 1800,
    }
    if use_regex is not None:
        pairlist_config["use_regex"] = use_regex
    default_conf["pairlists"] = [pairlist_config]

    selected_markets = {
        pair: markets[pair] for pair in ["ETH/USDT", "TKN/USDT", "XRP/USDT", "ETH/BTC"]
    }
    selected_markets["TKN/USDT"]["active"] = False
    exchange = get_patched_exchange(mocker, default_conf, mock_markets=selected_markets)

    response = MagicMock()
    response.headers = {"content-type": "application/json"}
    response.elapsed.total_seconds.return_value = 0.01
    response.json.return_value = {
        "pairs": [".*/USDT", "eth/usdt", "tkn/usdt", "eth/btc", "missing/usdt"]
    }
    request = mocker.patch(
        "freqtrade.plugins.pairlist.RemotePairList.requests.get", return_value=response
    )
    manager = PairListManager(exchange, default_conf)
    remote_pairlist = manager._pairlist_handlers[0]

    assert RemotePairList.available_parameters()["use_regex"]["default"] is True
    first_result = remote_pairlist.gen_pairlist({})
    assert first_result == expected
    first_result.clear()

    cached_result = remote_pairlist.gen_pairlist({})
    assert cached_result == expected
    cached_result.clear()
    assert remote_pairlist.gen_pairlist({}) == expected
    request.assert_called_once()
