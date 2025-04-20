```
usage: freqtrade backtesting-show [-h] [-v] [--no-color] [--logfile FILE] [-V]
                                  [-c PATH] [-d PATH] [--userdir PATH]
                                  [--export-filename PATH] [--show-pair-list]
                                  [--breakdown {day,week,month,year} [{day,week,month,year} ...]]

options:
  -h, --help            show this help message and exit
  --export-filename PATH, --backtest-filename PATH
                        Use this filename for backtest results.Requires
                        `--export` to be set as well. Example: `--export-filen
                        ame=user_data/backtest_results/backtest_today.json`
  --show-pair-list      Show backtesting pairlist sorted by profit.
  --breakdown {day,week,month,year} [{day,week,month,year} ...]
                        Show backtesting breakdown per [day, week, month,
                        year].

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
