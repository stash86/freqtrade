import timeit
from collections import defaultdict


# Simulate trade performance data
def generate_trade_data(num_trades):
    """Generate simulated trade performance data"""
    data = []
    enter_tags = ["buy_signal_1", "buy_signal_2", "buy_signal_3", None]
    exit_reasons = ["roi", "stoploss", "sell_signal", None]

    for i in range(num_trades):
        enter_tag = enter_tags[i % len(enter_tags)]
        exit_reason = exit_reasons[(i // len(enter_tags)) % len(exit_reasons)]
        data.append(
            {
                "enter_tag": enter_tag,
                "exit_reason": exit_reason,
                "profit": 0.001 * (i % 10),
                "profit_abs": 10.0 * (i % 5),
                "count": 1,
            }
        )
    return data


# Method 1: Current implementation (O(n²) with linear search)
def mix_tag_performance_linear(trades_data):
    resp = []
    for trade in trades_data:
        enter_tag = trade["enter_tag"] if trade["enter_tag"] is not None else "Other"
        exit_reason = trade["exit_reason"] if trade["exit_reason"] is not None else "Other"
        mix_tag = enter_tag + " " + exit_reason

        # Linear search through resp
        for item in resp:
            if item["mix_tag"] == mix_tag:
                item["profit_pct"] = round(trade["profit"] + item["profit_ratio"] * 100, 2)
                item["profit_ratio"] = trade["profit"] + item["profit_ratio"]
                item["profit_abs"] = trade["profit_abs"] + item["profit_abs"]
                item["count"] = 1 + item["count"]
                break
        else:
            resp.append(
                {
                    "mix_tag": mix_tag,
                    "profit_ratio": trade["profit"],
                    "profit_pct": round(trade["profit"] * 100, 2),
                    "profit_abs": trade["profit_abs"],
                    "count": trade["count"],
                }
            )
    return resp


# Method 2: Optimized (O(n) with dict lookup)
def mix_tag_performance_dict(trades_data):
    resp_dict = {}
    for trade in trades_data:
        enter_tag = trade["enter_tag"] if trade["enter_tag"] is not None else "Other"
        exit_reason = trade["exit_reason"] if trade["exit_reason"] is not None else "Other"
        mix_tag = enter_tag + " " + exit_reason

        if mix_tag in resp_dict:
            item = resp_dict[mix_tag]
            item["profit_pct"] = round(trade["profit"] + item["profit_ratio"] * 100, 2)
            item["profit_ratio"] = trade["profit"] + item["profit_ratio"]
            item["profit_abs"] = trade["profit_abs"] + item["profit_abs"]
            item["count"] = 1 + item["count"]
        else:
            resp_dict[mix_tag] = {
                "mix_tag": mix_tag,
                "profit_ratio": trade["profit"],
                "profit_pct": round(trade["profit"] * 100, 2),
                "profit_abs": trade["profit_abs"],
                "count": trade["count"],
            }
    return list(resp_dict.values())


# Test with different data sizes
test_sizes = [100, 500, 1000]

print(f"{'Trades':<10} {'Method':<25} {'Best of 3':>10}  {'per call':>12}")
print("-" * 60)

for size in test_sizes:
    trades_data = generate_trade_data(size)

    # Verify both produce same result
    result_linear = mix_tag_performance_linear(trades_data)
    result_dict = mix_tag_performance_dict(trades_data)
    assert len(result_linear) == len(result_dict), f"Size mismatch for {size} trades"

    # Benchmark linear approach
    t_linear = min(
        timeit.repeat(
            "mix_tag_performance_linear(trades_data)",
            globals={
                "mix_tag_performance_linear": mix_tag_performance_linear,
                "trades_data": trades_data,
            },
            number=100,
            repeat=3,
        )
    )

    # Benchmark dict approach
    t_dict = min(
        timeit.repeat(
            "mix_tag_performance_dict(trades_data)",
            globals={
                "mix_tag_performance_dict": mix_tag_performance_dict,
                "trades_data": trades_data,
            },
            number=100,
            repeat=3,
        )
    )

    print(
        f"{size:<10} {'Linear search (O(n²))':<25} {t_linear:>9.3f}s  {t_linear / 100 * 1e3:>10.2f} ms"
    )
    print(f"{'':<10} {'Dict lookup (O(n))':<25} {t_dict:>9.3f}s  {t_dict / 100 * 1e3:>10.2f} ms")
    print(f"{'':<10} {'Speedup':<25} {t_linear / t_dict:>9.2f}x")
    print()
