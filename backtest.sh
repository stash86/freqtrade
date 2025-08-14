rm user_data/backtest_results/*
#freqtrade backtesting --config user_data/other_configs/config-static-btc-futures.json --strategy-path ../freqtrade_strategies/ --timerange=20200201- --strategy Kedondong_btc_15m --fee 0.001 --stake-amount 500 --dry-run-wallet 200000 --cache none --breakdown month

#freqtrade backtesting --config user_data/other_configs/config-static-usdt-futures-mc-cg-combined.json --strategy-path ../freqtrade_strategies/ --timerange=20250101- --strategy Noken_1x_30m_1f1_new --fee 0.001 --stake-amount 500 --dry-run-wallet 200000 --cache none --breakdown month

# freqtrade lookahead-analysis --config user_data/other_configs/config-static-btc-futures.json --strategy-path ../freqtrade_strategies/ --timerange=20220201- --strategy Kasuari_btc_15m_1c_new --fee 0.001 --targeted-trade-amount 100

freqtrade recursive-analysis --config user_data/other_configs/config-static-btc-futures.json --strategy-path ../freqtrade_strategies/ --timerange=20220301-20250501 --strategy BBRSITV_15m
