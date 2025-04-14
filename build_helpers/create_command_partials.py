import subprocess  # noqa: S404, RUF100
from pathlib import Path


subcommands = [
    "trade",
    "create-userdir",
    "new-config",
    "show-config",
    "new-strategy",
    "download-data",
    "convert-data",
    "convert-trade-data",
    "trades-to-ohlcv",
    "list-data",
    "backtesting",
    "backtesting-show",
    "backtesting-analysis",
    "edge",
    "hyperopt",
    "hyperopt-list",
    "hyperopt-show",
    "list-exchanges",
    "list-markets",
    "list-pairs",
    "list-strategies",
    "list-hyperoptloss",
    "list-freqaimodels",
    "list-timeframes",
    "show-trades",
    "test-pairlist",
    "convert-db",
    "install-ui",
    "plot-dataframe",
    "plot-profit",
    "webserver",
    "strategy-updater",
    "lookahead-analysis",
    "recursive-analysis",
]

result = subprocess.run(["freqtrade", "--help"], capture_output=True, text=True)

with Path("docs/commands/main.md").open("w") as f:
    f.write(f"```\n{result.stdout}\n```\n")


for command in subcommands:
    print(f"Running for {command}")
    result = subprocess.run(["freqtrade", command, "--help"], capture_output=True, text=True)

    with Path(f"docs/commands/{command}.md").open("w") as f:
        f.write(f"```\n{result.stdout}\n```\n")
