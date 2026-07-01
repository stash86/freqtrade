import timeit


# Simulate an Order object
class MockOrder:
    def __init__(self, order_side, is_open, filled, status):
        self.ft_order_side = order_side
        self.ft_is_open = is_open
        self.filled = filled
        self.status = status


# Constants
NON_OPEN_EXCHANGE_STATES = {"closed", "canceled", "expired"}
ENTRY_SIDE = "buy"

# Create test orders (100 orders, mix of states)
orders = [
    MockOrder(
        order_side="buy" if i % 3 == 0 else "sell",
        is_open=(i % 10 < 3),
        filled=(i % 10 >= 3),
        status="closed" if i % 10 >= 3 else "open",
    )
    for i in range(100)
]


# Method 1: Manual loop (current implementation)
def count_manual(orders, order_side):
    count = 0
    for o in orders:
        if (
            ((o.ft_order_side == order_side) or (order_side is None))
            and o.ft_is_open is False
            and o.filled
            and o.status in NON_OPEN_EXCHANGE_STATES
        ):
            count += 1
    return count


# Method 2: Using sum() with generator
def count_sum(orders, order_side):
    return sum(
        1
        for o in orders
        if (
            ((o.ft_order_side == order_side) or (order_side is None))
            and o.ft_is_open is False
            and o.filled
            and o.status in NON_OPEN_EXCHANGE_STATES
        )
    )


# Method 3: Using list comprehension + len()
def count_list(orders, order_side):
    return len(
        [
            o
            for o in orders
            if (
                ((o.ft_order_side == order_side) or (order_side is None))
                and o.ft_is_open is False
                and o.filled
                and o.status in NON_OPEN_EXCHANGE_STATES
            )
        ]
    )


# Verify all produce same result
assert count_manual(orders, ENTRY_SIDE) == count_sum(orders, ENTRY_SIDE)
assert count_manual(orders, ENTRY_SIDE) == count_list(orders, ENTRY_SIDE)
print(f"✓ All methods return same result: {count_manual(orders, ENTRY_SIDE)}\n")

# Benchmark
N = 1_000_000

globals_dict = {
    "count_manual": count_manual,
    "count_sum": count_sum,
    "count_list": count_list,
    "orders": orders,
    "order_side": ENTRY_SIDE,
    "NON_OPEN_EXCHANGE_STATES": NON_OPEN_EXCHANGE_STATES,
}

t_manual = min(
    timeit.repeat(
        "count_manual(orders, order_side)",
        globals=globals_dict,
        number=N,
        repeat=5,
    )
)

t_sum = min(
    timeit.repeat(
        "count_sum(orders, order_side)",
        globals=globals_dict,
        number=N,
        repeat=5,
    )
)

t_list = min(
    timeit.repeat(
        "count_list(orders, order_side)",
        globals=globals_dict,
        number=N,
        repeat=5,
    )
)

print(f"{'Method':<30} {'Best of 5':>10}  {'per call':>10}")
print("-" * 55)
print(f"{'Manual loop':<30} {t_manual:>9.3f}s  {t_manual / N * 1e6:>8.2f} µs")
print(f"{'sum() with generator':<30} {t_sum:>9.3f}s  {t_sum / N * 1e6:>8.2f} µs")
print(f"{'list comprehension + len()':<30} {t_list:>9.3f}s  {t_list / N * 1e6:>8.2f} µs")
print(f"\nSpeedup (sum vs manual):  {t_manual / t_sum:.3f}x")
print(f"Speedup (list vs manual): {t_manual / t_list:.3f}x")
print(f"Speedup (sum vs list):    {t_list / t_sum:.3f}x")
