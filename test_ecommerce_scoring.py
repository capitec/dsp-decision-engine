"""E-commerce product recommendation scoring pipeline using the decider framework.

Pipeline:
1. User engagement score  (clicks, views, time_on_page)
2. Product popularity score (total_purchases, avg_rating, review_count)
3. Price affinity match    (user_price_sensitivity vs product_price)
4. Final combined recommendation score
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import polars as pl
from decider.modules.functional import generate_from_functions


# ---------------------------------------------------------------------------
# Stage 1: User Engagement Score
# ---------------------------------------------------------------------------

def click_score(clicks: pl.Expr) -> pl.Expr:
    """Normalise click count to 0-1 using soft cap at 20."""
    return (clicks / 20.0).clip(0.0, 1.0)


def view_score(views: pl.Expr) -> pl.Expr:
    """Normalise view count to 0-1 using soft cap at 50."""
    return (views / 50.0).clip(0.0, 1.0)


def time_score(time_on_page: pl.Expr) -> pl.Expr:
    """Normalise time-on-page (seconds) to 0-1 using soft cap at 120 s."""
    return (time_on_page / 120.0).clip(0.0, 1.0)


def engagement_score(click_score: pl.Expr, view_score: pl.Expr, time_score: pl.Expr) -> pl.Expr:
    """Weighted combination of click, view and time sub-scores."""
    return (click_score * 0.4 + view_score * 0.3 + time_score * 0.3)


# ---------------------------------------------------------------------------
# Stage 2: Product Popularity Score
# ---------------------------------------------------------------------------

def purchase_score(total_purchases: pl.Expr) -> pl.Expr:
    """Normalise purchase count to 0-1 using soft cap at 500."""
    return (total_purchases / 500.0).clip(0.0, 1.0)


def rating_score(avg_rating: pl.Expr) -> pl.Expr:
    """Map a 1-5 star rating to 0-1."""
    return ((avg_rating - 1.0) / 4.0).clip(0.0, 1.0)


def review_score(review_count: pl.Expr) -> pl.Expr:
    """Normalise review count to 0-1 using soft cap at 200."""
    return (review_count / 200.0).clip(0.0, 1.0)


def popularity_score(purchase_score: pl.Expr, rating_score: pl.Expr, review_score: pl.Expr) -> pl.Expr:
    """Weighted combination of purchase, rating and review sub-scores."""
    return (purchase_score * 0.5 + rating_score * 0.35 + review_score * 0.15)


# ---------------------------------------------------------------------------
# Stage 3: Price Affinity Score
# ---------------------------------------------------------------------------

def price_affinity(user_price_ceiling: pl.Expr, product_price: pl.Expr) -> pl.Expr:
    """Score how well the product price fits the user's budget.

    Returns 1.0 when product_price <= user_price_ceiling,
    decaying toward 0 as the product price exceeds the ceiling.
    """
    ratio = product_price / user_price_ceiling.clip(lower_bound=0.01)
    return (1.0 - (ratio - 1.0).clip(lower_bound=0.0)).clip(lower_bound=0.0)


# ---------------------------------------------------------------------------
# Stage 4: Final Recommendation Score
# ---------------------------------------------------------------------------

def recommendation_score(
    engagement_score: pl.Expr,
    popularity_score: pl.Expr,
    price_affinity: pl.Expr,
) -> pl.Expr:
    """Combine all sub-scores into a final 0-100 recommendation score."""
    raw = engagement_score * 0.35 + popularity_score * 0.40 + price_affinity * 0.25
    return (raw * 100.0).round(2)


# ---------------------------------------------------------------------------
# Build modules
# ---------------------------------------------------------------------------

EngagementModule = generate_from_functions(
    "engagement",
    click_score,
    view_score,
    time_score,
    engagement_score,
)

PopularityModule = generate_from_functions(
    "popularity",
    purchase_score,
    rating_score,
    review_score,
    popularity_score,
)

PriceAffinityModule = generate_from_functions(
    "price_affinity",
    price_affinity,
)

RecommendationModule = generate_from_functions(
    "recommendation",
    engagement_score,
    popularity_score,
    price_affinity,
    recommendation_score,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def make_test_df() -> pl.DataFrame:
    return pl.DataFrame({
        # User behaviour
        "clicks":              [0,    5,    15,   25],
        "views":               [2,    20,   40,   60],
        "time_on_page":        [10.0, 45.0, 90.0, 150.0],
        # Product attributes
        "total_purchases":     [10,   150,  400,  600],
        "avg_rating":          [2.5,  3.8,  4.5,  4.9],
        "review_count":        [5,    60,   180,  300],
        # Price
        "user_price_ceiling":  [50.0, 100.0, 200.0, 80.0],
        "product_price":       [45.0, 120.0, 180.0, 120.0],
    })


def test_engagement_score():
    print("\n=== Stage 1: User Engagement Score ===")
    module = EngagementModule(name="engagement")
    df = make_test_df()
    result = module.execute({"input": df})["input"].collect()

    print(result.select(["clicks", "views", "time_on_page",
                          "click_score", "view_score", "time_score", "engagement_score"]))

    assert "engagement_score" in result.columns
    scores = result["engagement_score"].to_list()
    assert all(0.0 <= s <= 1.0 for s in scores), f"Scores out of range: {scores}"
    # Row 0 has minimal activity — should score low
    assert scores[0] < scores[2], "High-activity row should outscore low-activity row"
    print("Engagement scores:", [round(s, 3) for s in scores])
    print("PASS")


def test_popularity_score():
    print("\n=== Stage 2: Product Popularity Score ===")
    module = PopularityModule(name="popularity")
    df = make_test_df()
    result = module.execute({"input": df})["input"].collect()

    print(result.select(["total_purchases", "avg_rating", "review_count",
                          "purchase_score", "rating_score", "review_score", "popularity_score"]))

    assert "popularity_score" in result.columns
    scores = result["popularity_score"].to_list()
    assert all(0.0 <= s <= 1.0 for s in scores), f"Scores out of range: {scores}"
    # Row 3 has most purchases (capped) and best rating
    assert scores[3] > scores[0], "Popular product should outscore unpopular one"
    print("Popularity scores:", [round(s, 3) for s in scores])
    print("PASS")


def test_price_affinity():
    print("\n=== Stage 3: Price Affinity Score ===")
    module = PriceAffinityModule(name="price_affinity")
    df = make_test_df()
    result = module.execute({"input": df})["input"].collect()

    print(result.select(["user_price_ceiling", "product_price", "price_affinity"]))

    assert "price_affinity" in result.columns
    affinities = result["price_affinity"].to_list()
    # Row 0: product (45) < ceiling (50) → affinity = 1.0
    assert affinities[0] == 1.0, f"Expected 1.0 for in-budget product, got {affinities[0]}"
    # Row 1: product (120) > ceiling (100) → affinity < 1.0
    assert affinities[1] < 1.0, f"Expected <1.0 for over-budget product, got {affinities[1]}"
    # Row 3: product (120) > ceiling (80) → ratio 1.5 → affinity = 1 - 0.5 = 0.5
    assert 0.0 < affinities[3] < 1.0, f"Expected partial affinity for over-budget product, got {affinities[3]}"
    # Row 3 affinity should be worse than row 0 (in-budget)
    assert affinities[3] < affinities[0], "Over-budget product should have lower affinity than in-budget"
    print("Price affinity scores:", [round(a, 3) for a in affinities])
    print("PASS")


def test_full_pipeline():
    print("\n=== Stage 4: Full Recommendation Pipeline ===")

    # Run stages sequentially, accumulating columns
    df = make_test_df()

    eng_module  = EngagementModule(name="engagement")
    pop_module  = PopularityModule(name="popularity")
    price_module = PriceAffinityModule(name="price_affinity")
    rec_module  = RecommendationModule(name="recommendation")

    # Each module adds its columns to the dataframe
    df = eng_module.execute({"input": df})["input"].collect()
    df = pop_module.execute({"input": df})["input"].collect()
    df = price_module.execute({"input": df})["input"].collect()
    df = rec_module.execute({"input": df})["input"].collect()

    print(df.select([
        "clicks", "avg_rating", "product_price", "user_price_ceiling",
        "engagement_score", "popularity_score", "price_affinity",
        "recommendation_score",
    ]))

    assert "recommendation_score" in df.columns
    scores = df["recommendation_score"].to_list()
    assert all(0.0 <= s <= 100.0 for s in scores), f"Final scores out of 0-100 range: {scores}"
    print("Final recommendation scores:", scores)

    # Row 2 should rank well (decent engagement, high popularity, within budget)
    # Row 3 is over budget so should be penalised despite popularity
    assert scores[2] > scores[0], "Active+popular product should outscore inactive+unpopular one"
    print("PASS")


def test_compiled_reuse():
    print("\n=== Bonus: Compiled Plan Reuse ===")
    module = EngagementModule(name="engagement")
    compiled = module.compile()

    batch1 = pl.DataFrame({"clicks": [1], "views": [5], "time_on_page": [30.0]})
    batch2 = pl.DataFrame({"clicks": [10], "views": [30], "time_on_page": [90.0]})

    r1 = compiled.execute({"input": batch1})["input"].collect()
    r2 = compiled.execute({"input": batch2})["input"].collect()

    assert r1["engagement_score"][0] < r2["engagement_score"][0], \
        "Higher-activity batch should produce higher engagement score"
    print(f"Batch 1 engagement: {r1['engagement_score'][0]:.3f}")
    print(f"Batch 2 engagement: {r2['engagement_score'][0]:.3f}")
    print("PASS")


if __name__ == "__main__":
    test_engagement_score()
    test_popularity_score()
    test_price_affinity()
    test_full_pipeline()
    test_compiled_reuse()

    print("\n" + "=" * 60)
    print("All e-commerce scoring tests passed!")
    print("=" * 60)
