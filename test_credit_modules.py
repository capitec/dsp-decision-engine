"""Test that existing credit modules work with the new executor architecture."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import polars as pl
from decider.modules.credit.scorecard.module import ScoredVariable, ScoreCard
from decider.modules.credit.scorecard.impl import BoundBin, DefaultBin


def test_scored_variable_with_new_executor():
    """Test ScoredVariable compiles and executes with new executor."""
    print("\n=== Test: ScoredVariable with New Executor ===")

    # Create a scored variable
    scored_var = ScoredVariable(
        name="income_score",
        type="scored",
        variable_name="income",
        bins=[
            BoundBin(lower_bound=None, upper_bound=30000, value=0),
            BoundBin(lower_bound=30000, upper_bound=60000, value=50),
            BoundBin(lower_bound=60000, upper_bound=None, value=100),
        ],
        default=DefaultBin(value=-100),
        value_output_name="income_score"
    )

    # Create test data
    df = pl.DataFrame({
        "income": [25000.0, 45000.0, 75000.0, 100000.0]
    })

    # Compile with new executor
    compiled = scored_var.compile()
    print(f"Compiled type: {type(compiled).__name__}")
    print(f"Expression groups: {len(compiled.expression_groups)}")

    # Execute
    result = compiled.execute({"input": df})
    result_df = result.collect()

    print(f"Result:\n{result_df}")

    # Verify scores
    assert "income_score" in result_df.columns
    scores = result_df["income_score"].to_list()
    expected = [0.0, 50.0, 100.0, 100.0]  # Based on bins
    assert scores == expected, f"Expected {expected}, got {scores}"

    print("✓ ScoredVariable works with new executor")


def test_scorecard_with_new_executor():
    """Test ScoreCard compiles and executes."""
    print("\n=== Test: ScoreCard with New Executor ===")

    # Create scored variables
    income_var = ScoredVariable(
        name="income",
        type="scored",
        variable_name="income",
        bins=[
            BoundBin(lower_bound=None, upper_bound=50000, value=0),
            BoundBin(lower_bound=50000, upper_bound=None, value=100),
        ],
        default=DefaultBin(value=-100),
        value_output_name="income_score"
    )

    age_var = ScoredVariable(
        name="age",
        type="scored",
        variable_name="age",
        bins=[
            BoundBin(lower_bound=None, upper_bound=25, value=0),
            BoundBin(lower_bound=25, upper_bound=50, value=50),
            BoundBin(lower_bound=50, upper_bound=None, value=100),
        ],
        default=DefaultBin(value=-100),
        value_output_name="age_score"
    )

    # Create scorecard
    scorecard = ScoreCard(
        name="credit_scorecard",
        type="scorecard",
        variables=[income_var, age_var],
        output_name="total_score"
    )

    # Test data
    df = pl.DataFrame({
        "income": [40000.0, 60000.0, 80000.0],
        "age": [20.0, 30.0, 55.0]
    })

    # Compile and execute
    compiled = scorecard.compile()
    result = compiled.execute({"input": df})
    result_df = result.collect()

    print(f"Result:\n{result_df}")

    # Verify individual scores
    assert "income_score" in result_df.columns
    assert "age_score" in result_df.columns
    assert "total_score" in result_df.columns

    # Check totals (sum of component scores)
    expected_totals = [0.0, 150.0, 200.0]  # (0+0), (100+50), (100+100)
    actual_totals = result_df["total_score"].to_list()
    assert actual_totals == expected_totals, f"Expected {expected_totals}, got {actual_totals}"

    print("✓ ScoreCard works with new executor")


def test_convenience_execute():
    """Test that convenience execute() method works."""
    print("\n=== Test: Convenience Execute ===")

    scored_var = ScoredVariable(
        name="amount_score",
        type="scored",
        variable_name="amount",
        bins=[
            BoundBin(lower_bound=None, upper_bound=100, value=10),
            BoundBin(lower_bound=100, upper_bound=None, value=90),
        ],
        default=DefaultBin(value=0),
        value_output_name="amount_score"
    )

    df = pl.DataFrame({"amount": [50.0, 150.0, 250.0]})

    # Use convenience method — returns LazyFrame by default
    result = scored_var.execute({"input": df})
    result_df = result.collect()

    print(f"Result:\n{result_df}")

    assert "amount_score" in result_df.columns
    assert result_df["amount_score"].to_list() == [10.0, 90.0, 90.0]

    print("✓ Convenience execute works")


if __name__ == "__main__":
    test_scored_variable_with_new_executor()
    test_scorecard_with_new_executor()
    test_convenience_execute()

    print("\n" + "="*60)
    print("All credit module tests passed! ✓")
    print("="*60)
    print("\nThe existing credit modules work perfectly with the new")
    print("executor architecture because they implement expand_nodes()!")
