import subprocess  # noqa: S404, RUF100
import sys
from pathlib import Path


def _write_partial_file(filename: str, content: str):
    with Path(filename).open("w") as f:
        f.write(f"``` output\n{content}\n```\n")


def extract_command_partials():
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

    _write_partial_file("docs/commands/main.md", result.stdout)

    for command in subcommands:
        print(f"Running for {command}")
        result = subprocess.run(["freqtrade", command, "--help"], capture_output=True, text=True)

        _write_partial_file(f"docs/commands/{command}.md", result.stdout)

    print("Running for freqtrade-client")
    result_client = subprocess.run(["freqtrade-client", "--show"], capture_output=True, text=True)

    _write_partial_file("docs/commands/freqtrade-client.md", result_client.stdout)


if __name__ == "__main__":
    if sys.version_info < (3, 13):  # pragma: no cover
        sys.exit(
            "argparse output changed in Python 3.13+. "
            "To keep command partials up to date, please run this script with Python 3.13+."
        )
    extract_command_partials()
