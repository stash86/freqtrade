```
usage: freqtrade list-exchanges [-h] [-v] [--no-color] [--logfile FILE] [-V]
                                [-c PATH] [-d PATH] [--userdir PATH] [-1] [-a]
                                [--trading-mode {spot,margin,futures}]
                                [--dex-exchanges]

options:
  -h, --help            show this help message and exit
  -1, --one-column      Print output in one column.
  -a, --all             Print all exchanges known to the ccxt library.
  --trading-mode {spot,margin,futures}, --tradingmode {spot,margin,futures}
                        Select Trading mode
  --dex-exchanges       Print only DEX exchanges.

Common arguments:
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  --no-color            Disable colorization of hyperopt results. May be
                        useful if you are redirecting output to a file.
  --logfile FILE, --log-file FILE
                        Log to the file specified. Special values are:
                        'syslog', 'journald'. See the documentation for more
                        details.
  -V, --version         show program's version number and exit
  -c PATH, --config PATH
                        Specify configuration file (default:
                        `userdir/config.json` or `config.json` whichever
                        exists). Multiple --config options may be used. Can be
                        set to `-` to read config from stdin.
  -d PATH, --datadir PATH, --data-dir PATH
                        Path to the base directory of the exchange with
                        historical backtesting data. To see futures data, use
                        trading-mode additionally.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.

```
