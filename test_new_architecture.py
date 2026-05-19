"""Test the new executor-based architecture.

This test verifies:
1. Module.compile() returns a CompiledDag
2. CompiledDag.execute() can be reused multiple times
3. Module.execute() works as a convenience method
4. Custom executors can be passed in
5. Global executor settings work
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import polars as pl
from decider.modules.functional import generate_from_functions
from decider.settings import get_default_executor, set_default_executor, reset_default_executor


def test_basic_compile_and_execute():
    """Test basic compilation and execution."""
    print("\n=== Test 1: Basic Compile and Execute ===")

    def score(amount: pl.Expr) -> pl.Expr:
        return amount * 100

    # Generate module
    Module = generate_from_functions("scorer", score)
    module = Module(name="scorer")

    # Compile
    compiled = module.compile()
    print(f"Compiled type: {type(compiled).__name__}")
    print(f"Number of expression groups: {len(compiled.expression_groups)}")

    # Execute
    df = pl.DataFrame({"amount": [1.0, 2.0, 3.0]})
    result = compiled.execute({"input": df})

    # Verify (compiled.execute returns LazyFrame by default)
    result_df = result.collect()
    assert "score" in result_df.columns
    assert result_df["score"].to_list() == [100.0, 200.0, 300.0]

    print("✓ Basic compile and execute works")


def test_reuse_compiled():
    """Test that compiled plan can be reused multiple times."""
    print("\n=== Test 2: Reuse Compiled Plan ===")

    def doubled(amount: pl.Expr) -> pl.Expr:
        return amount * 2

    Module = generate_from_functions("doubler", doubled)
    module = Module(name="doubler")

    # Compile once
    compiled = module.compile()

    # Execute multiple times with different data
    df1 = pl.DataFrame({"amount": [1.0, 2.0]})
    df2 = pl.DataFrame({"amount": [10.0, 20.0]})
    df3 = pl.DataFrame({"amount": [100.0, 200.0]})

    result1 = compiled.execute({"input": df1}).collect()
    result2 = compiled.execute({"input": df2}).collect()
    result3 = compiled.execute({"input": df3}).collect()

    assert result1["doubled"].to_list() == [2.0, 4.0]
    assert result2["doubled"].to_list() == [20.0, 40.0]
    assert result3["doubled"].to_list() == [200.0, 400.0]

    print("✓ Compiled plan can be reused")


def test_convenience_execute():
    """Test that module.execute() works as a convenience method."""
    print("\n=== Test 3: Convenience Execute Method ===")

    def squared(amount: pl.Expr) -> pl.Expr:
        return amount ** 2

    Module = generate_from_functions("squarer", squared)
    module = Module(name="squarer")

    # Use execute directly (compiles internally) — returns LazyFrame by default
    df = pl.DataFrame({"amount": [2.0, 3.0, 4.0]})
    result = module.execute({"input": df})

    result_df = result.collect()
    assert "squared" in result_df.columns
    assert result_df["squared"].to_list() == [4.0, 9.0, 16.0]

    print("✓ Convenience execute() works")


def test_with_dependencies():
    """Test compilation with dependent expressions."""
    print("\n=== Test 4: Dependent Expressions ===")

    def amount_mean(amount: pl.Expr) -> pl.Expr:
        return amount.mean()

    def amount_centered(amount: pl.Expr, amount_mean: pl.Expr) -> pl.Expr:
        return amount - amount_mean

    Module = generate_from_functions("normalizer", amount_mean, amount_centered)
    module = Module(name="normalizer")

    # Compile
    compiled = module.compile()
    print(f"Expression groups: {len(compiled.expression_groups)}")

    # Execute
    df = pl.DataFrame({"amount": [100.0, 200.0, 300.0]})
    result = compiled.execute({"input": df}).collect()

    print(f"Result:\n{result}")

    assert "amount_mean" in result.columns
    assert "amount_centered" in result.columns
    assert abs(result["amount_mean"][0] - 200.0) < 0.01
    assert result["amount_centered"].to_list() == [-100.0, 0.0, 100.0]

    print("✓ Dependent expressions work correctly")


def test_global_executor_settings():
    """Test that global executor settings work."""
    print("\n=== Test 5: Global Executor Settings ===")

    # Reset to default
    reset_default_executor()

    # Get default
    executor = get_default_executor()
    print(f"Default executor: {type(executor).__name__}")

    from decider.executor import SimpleExecutor
    assert isinstance(executor, SimpleExecutor)

    # Set a new default
    custom_executor = SimpleExecutor()
    set_default_executor(custom_executor)

    # Verify it's used
    new_executor = get_default_executor()
    assert new_executor is custom_executor

    # Reset
    reset_default_executor()

    print("✓ Global executor settings work")


def test_debug_mode():
    """Test debug mode output."""
    print("\n=== Test 6: Debug Mode ===")

    def score(amount: pl.Expr) -> pl.Expr:
        return amount * 10

    Module = generate_from_functions("scorer", score)
    module = Module(name="scorer")

    df = pl.DataFrame({"amount": [1.0, 2.0, 3.0]})

    print("\nExecuting with debug=True:")
    print("-" * 60)
    result = module.execute({"input": df}, debug=True)
    print("-" * 60)

    result_df = result.collect()
    assert "score" in result_df.columns

    print("✓ Debug mode works")


def test_caching_scenario():
    """Test realistic caching scenario for realtime execution."""
    print("\n=== Test 7: Realtime Caching Scenario ===")

    import time

    def feature_a(amount: pl.Expr) -> pl.Expr:
        return amount * 2

    def feature_b(amount: pl.Expr) -> pl.Expr:
        return amount.log()

    def score(feature_a: pl.Expr, feature_b: pl.Expr) -> pl.Expr:
        return feature_a + feature_b * 10

    Module = generate_from_functions("scorer", feature_a, feature_b, score)
    module = Module(name="scorer")

    # Compile once (expensive operation)
    start = time.perf_counter()
    compiled = module.compile()
    compile_time = time.perf_counter() - start
    print(f"Compile time: {compile_time*1000:.2f}ms")

    # Execute many times (hot path)
    batches = [
        pl.DataFrame({"amount": [float(i), float(i+1), float(i+2)]})
        for i in range(1, 100, 3)
    ]

    start = time.perf_counter()
    for batch in batches:
        result = compiled.execute({"input": batch})
    execute_time = time.perf_counter() - start
    avg_per_batch = execute_time / len(batches) * 1000

    print(f"Total execute time for {len(batches)} batches: {execute_time*1000:.2f}ms")
    print(f"Average per batch: {avg_per_batch:.2f}ms")

    # Verify last result
    last_result = result.collect()
    assert all(col in last_result.columns for col in ["feature_a", "feature_b", "score"])

    print("✓ Caching scenario works efficiently")


if __name__ == "__main__":
    test_basic_compile_and_execute()
    test_reuse_compiled()
    test_convenience_execute()
    test_with_dependencies()
    test_global_executor_settings()
    test_debug_mode()
    test_caching_scenario()

    print("\n" + "="*60)
    print("All new architecture tests passed! ✓")
    print("="*60)
