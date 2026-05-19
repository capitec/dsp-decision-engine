"""Test JoinModule and frame operations."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import polars as pl
from decider.modules.primitives.join import JoinModule
from decider.modules.functional import generate_from_functions


def test_basic_join():
    """Test a simple left join."""
    print("\n=== Test 1: Basic Left Join ===")

    transactions = pl.DataFrame({
        "txn_id": [1, 2, 3, 4],
        "user_id": [101, 102, 101, 103],
        "amount": [100.0, 200.0, 150.0, 300.0]
    })

    users = pl.DataFrame({
        "user_id": [101, 102, 103],
        "name": ["Alice", "Bob", "Charlie"],
        "tier": ["gold", "silver", "bronze"]
    })

    join = JoinModule(
        name="user_txn_join",
        left="transactions",
        right="users",
        on="user_id",
        how="left",
        output_frame="enriched"
    )

    # output_frames=["enriched"] to retrieve the named frame
    result = join.execute(
        {"transactions": transactions, "users": users},
        output_frames=["enriched"]
    )

    assert "enriched" in result
    enriched = result["enriched"].collect()

    print(f"Enriched data:\n{enriched}")

    assert len(enriched) == 4
    assert "name" in enriched.columns
    assert "tier" in enriched.columns
    assert enriched["name"].to_list() == ["Alice", "Bob", "Alice", "Charlie"]

    print("✓ Basic join works")


def test_join_with_expressions():
    """Test joining data and then applying expressions to the joined result."""
    print("\n=== Test 2: Join + Expressions ===")

    transactions = pl.DataFrame({
        "txn_id": [1, 2, 3],
        "user_id": [101, 102, 101],
        "amount": [100.0, 200.0, 150.0]
    })

    users = pl.DataFrame({
        "user_id": [101, 102],
        "tier_multiplier": [1.5, 1.2]
    })

    join = JoinModule(
        name="join",
        left="transactions",
        right="users",
        on="user_id",
        output_frame="enriched"
    )

    def adjusted_amount(amount: pl.Expr, tier_multiplier: pl.Expr) -> pl.Expr:
        return amount * tier_multiplier

    ExprModule = generate_from_functions("scorer", adjusted_amount)
    expr_module = ExprModule(name="scorer")

    # Compose: join | expr_module
    # The pipeline threads the join's output frame ("enriched") into the expr module
    pipeline = join | expr_module
    result = pipeline.execute(
        {"transactions": transactions, "users": users}
    )

    final = result.collect()

    print(f"Final result:\n{final}")

    assert "adjusted_amount" in final.columns
    # txn1: 100 * 1.5 = 150, txn2: 200 * 1.2 = 240, txn3: 150 * 1.5 = 225
    expected = [150.0, 240.0, 225.0]
    assert final["adjusted_amount"].to_list() == expected

    print("✓ Join + expressions work")


def test_inner_join():
    """Test inner join behavior."""
    print("\n=== Test 3: Inner Join ===")

    left_df = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "value": ["a", "b", "c", "d"]
    })

    right_df = pl.DataFrame({
        "id": [2, 3, 5],
        "score": [10, 20, 30]
    })

    join = JoinModule(
        name="inner_join",
        left="left",
        right="right",
        on="id",
        how="inner",
        output_frame="result"
    )

    result = join.execute(
        {"left": left_df, "right": right_df},
        output_frames=["result"]
    )
    result_df = result["result"].collect()

    print(f"Inner join result:\n{result_df}")

    assert len(result_df) == 2
    assert result_df["id"].to_list() == [2, 3]
    assert result_df["value"].to_list() == ["b", "c"]
    assert result_df["score"].to_list() == [10, 20]

    print("✓ Inner join works correctly")


def test_multiple_joins():
    """Test chaining multiple joins via pipeline."""
    print("\n=== Test 4: Multiple Joins ===")

    transactions = pl.DataFrame({
        "txn_id": [1, 2, 3],
        "user_id": [101, 102, 101],
        "product_id": [501, 502, 501]
    })

    users = pl.DataFrame({
        "user_id": [101, 102],
        "user_name": ["Alice", "Bob"]
    })

    products = pl.DataFrame({
        "product_id": [501, 502],
        "product_name": ["Widget", "Gadget"],
        "price": [10.0, 20.0]
    })

    join1 = JoinModule(
        name="user_join",
        left="transactions",
        right="users",
        on="user_id",
        output_frame="txn_with_users"
    )

    join2 = JoinModule(
        name="product_join",
        left="txn_with_users",
        right="products",
        on="product_id",
        output_frame="fully_enriched"
    )

    pipeline = join1 | join2
    result = pipeline.execute(
        {"transactions": transactions, "users": users, "products": products},
        output_frames=["fully_enriched"]
    )

    final = result["fully_enriched"].collect()

    print(f"Fully enriched data:\n{final}")

    assert "user_name" in final.columns
    assert "product_name" in final.columns
    assert "price" in final.columns
    assert len(final) == 3

    print("✓ Multiple joins work")


def test_compile_join():
    """Test that join modules can be compiled for reuse."""
    print("\n=== Test 5: Compile Join for Reuse ===")

    join = JoinModule(
        name="join",
        left="left",
        right="right",
        on="key",
        output_frame="joined"
    )

    compiled = join.compile()

    print(f"Compiled type: {type(compiled).__name__}")
    print(f"Frame operations: {len(compiled.frame_operations)}")

    data_sets = [
        (
            pl.DataFrame({"key": [1, 2], "val_left": ["a", "b"]}),
            pl.DataFrame({"key": [1, 2], "val_right": ["x", "y"]})
        ),
        (
            pl.DataFrame({"key": [10, 20], "val_left": ["c", "d"]}),
            pl.DataFrame({"key": [10, 20], "val_right": ["z", "w"]})
        ),
    ]

    for i, (left_df, right_df) in enumerate(data_sets):
        result = compiled.execute({
            "left": left_df,
            "right": right_df
        })

        joined = result.collect()
        print(f"Execution {i+1}: {len(joined)} rows")
        assert len(joined) == 2
        assert "val_left" in joined.columns
        assert "val_right" in joined.columns

    print("✓ Compiled join can be reused")


if __name__ == "__main__":
    test_basic_join()
    test_join_with_expressions()
    test_inner_join()
    test_multiple_joins()
    test_compile_join()

    print("\n" + "="*60)
    print("All join tests passed! ✓")
    print("="*60)
