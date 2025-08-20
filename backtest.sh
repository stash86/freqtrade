rm user_data/backtest_results/*
freqtrade backtesting --config user_data/other_configs/config-static-btc-futures.json --strategy-path ../freqtrade_strategies/ --timerange=20200201- --strategy Sate_btc_30m_1a --fee 0.001 --stake-amount 500 --dry-run-wallet 200000 --cache none --breakdown month

#freqtrade backtesting --config user_data/other_configs/config-static-usdt-futures-mc-cg-combined.json --strategy-path ../freqtrade_strategies/ --timerange=20250101- --strategy Noken_1x_30m_1f1_new --fee 0.001 --stake-amount 500 --dry-run-wallet 200000 --cache none --breakdown month

#freqtrade lookahead-analysis --config user_data/other_configs/config-static-usdt-futures-mc-cg-combined.json --strategy-path ../freqtrade_strategies/ --timerange=20250201- --strategy Rsiqui --fee 0.001 --targeted-trade-amount 10

#freqtrade lookahead-analysis --config user_data/other_configs/config-static-btc-futures.json --strategy-path ../freqtrade_strategies/ --timerange=20220201- --strategy Rsiqui --fee 0.001 --targeted-trade-amount 10

#freqtrade recursive-analysis --config user_data/other_configs/config-static-btc-futures.json --strategy-path ../freqtrade_strategies/ --timerange=20220301-20250501 --strategy Rsiqui
