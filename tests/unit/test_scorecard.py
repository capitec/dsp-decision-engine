import pytest

from unittest.mock import patch, mock_open

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from spockflow.components import scorecard


@pytest.fixture(scope="module")
def pd_input_data():
    """Setup fixture to test PD calcs."""
    # Create the DataFrame with the values
    data = [(100, -16.700), (660, 2.708), (700, 4.094)]

    # Create the DataFrame using createDataFrame method
    yield pd.DataFrame(data, columns=["score", "log_odds_test_input"])


@pytest.fixture(scope="module")
def pd_scorecard_data() -> pd.DataFrame:
    """Setup fixture to test scorecard funcs."""
    # Create the DataFrame with the values
    data = [
        (1.0, "a", 0),
        (-1.0, "b", 1),
        (0.0, "c", 2),
        (0.1, "d", 1),
        (-0.1, "e", 2),
        (float("inf"), "f", 3),
        (float("nan"), "a" * 100_000, 2),
        (-float("inf"), "", 3),
        (None, None, 0),
    ]
    # Create the DataFrame using createDataFrame method
    yield pd.DataFrame(data, columns=["value", "category", "category_idx"])


@pytest.fixture(scope="module")
def pd_scorecard_result_data(pd_scorecard_data: "pd.DataFrame"):
    """Setup fixture to test adjustment modules."""
    # Set the data for each field
    score_data = {
        # "value": [float(v) for v in pd_scorecard_data["value"].values.tolist()],
        # "category": pd_scorecard_data["category"].values.tolist(),
        # "category_idx": pd_scorecard_data["category_idx"].values.tolist(),
        "TESTGRP_value": [1, 6, 7, 1, 3, 6, 5, 2, 5],
        "TESTPOINTS_value": [10.0, 60.0, 70.0, 10.0, 30.0, 60.0, 50.0, 20.0, 50.0],
        "TESTDESC_value": [
            ">=0",
            "default",
            "0",
            ">=0",
            "[-0.1, 0)",
            "default",
            "nan",
            "-inf",
            "nan",
        ],
        "TESTGRP_category": [1, 2, 2, 3, 3, 3, 0, 0, 3],
        "TESTPOINTS_category": [
            90.0,
            100.0,
            100.0,
            110.0,
            110.0,
            110.0,
            80.0,
            80.0,
            110.0,
        ],
        "TESTDESC_category": [
            "a",
            "b,c",
            "b,c",
            "d,e,f,None",
            "d,e,f,None",
            "d,e,f,None",
            "default",
            "default",
            "d,e,f,None",
        ],
        "TESTGRP_category_idx": [-1, 0, 1, 0, 1, 1, 1, 1, -1],
        "TESTPOINTS_category_idx": [
            -1.0,
            120.0,
            130.0,
            120.0,
            130.0,
            130.0,
            130.0,
            130.0,
            -1.0,
        ],
        "TESTDESC_category_idx": [
            None,
            "1",
            "2,3",
            "1",
            "2,3",
            "2,3",
            "2,3",
            "2,3",
            None,
        ],
    }
    score_data["TESTPOINTS_SUM"] = [
        sum(v)
        for v in zip(*[v for k, v in score_data.items() if k.startswith("TESTPOINTS_")])
    ]
    score_data["TESTPOINTS_SUM_PD_LOGODDS"] = [
        1.53242156,
        15.47238152,
        17.01270859,
        12.39172739,
        14.70221799,
        17.01270859,
        13.93205446,
        11.62156385,
        6.1534027627715195,
    ]
    score_data["TESTPOINTS_SUM_PD"] = [
        1.77639659e-01,
        1.90734827e-07,
        4.08765768e-08,
        4.15278427e-06,
        4.12009924e-07,
        4.08765768e-08,
        8.89990235e-07,
        8.97046697e-06,
        0.002121723094693717,
    ]
    # Create the DataFrame using createDataFrame method
    yield pd.DataFrame(score_data)


@pytest.fixture(scope="module")
def adjustment_scorecard_model():
    return (
        get_test_scorecard(
            score_scaling_params={
                "base_points": 100,
                "base_odds": 5,
                "pdo": 9,
            }
        )
        .add_criteria(scorecard.ScoreCriteria("value", "numerical"))
        .add_criteria(scorecard.ScoreCriteria("category", "numerical"))
        .add_criteria(scorecard.ScoreCriteria("category_idx", "numerical"))
    )


def test_log_odds_from_score(pd_input_data: "pd.DataFrame"):
    """test log odds."""
    base_points = 660
    base_odds = 15
    pdo = 20

    # Call the function on the DataFrame
    result = scorecard.log_odds_from_score(
        pd_input_data["score"], base_points, base_odds, pdo
    )

    # Extract the result from the DataFrame

    assert result.iloc[0] == pytest.approx(-16.700, abs=1e-3)
    assert result.iloc[1] == pytest.approx(2.708, abs=1e-3)
    assert result.iloc[2] == pytest.approx(4.094, abs=1e-3)


def test_probability_of_default_from_log_odds(pd_input_data):

    # Extract the result from the DataFrame
    result = scorecard.probability_of_default_from_log_odds(
        pd_input_data["log_odds_test_input"]
    )

    assert result.iloc[0] == pytest.approx(0.9999, abs=1e-3)
    assert result.iloc[1] == pytest.approx(0.0625, abs=1e-3)
    assert result.iloc[2] == pytest.approx(0.0163, abs=1e-3)


def get_test_scorecard(**kwargs) -> scorecard.ScoreCardModelV2:
    return scorecard.ScoreCard(
        bin_prefix="TESTGRP_",
        score_prefix="TESTPOINTS_",
        description_prefix="TESTDESC_",
        **kwargs,
    )


def test_numerical_range_scorecard(pd_scorecard_data: "pd.DataFrame"):
    scorecard_runner = get_test_scorecard().add_criteria(
        scorecard.ScoreCriteria("value", "numerical").add_range_score(
            0, "inf", 10, "positive"
        )
    )

    res = scorecard_runner.score_numerical_vars(pd_scorecard_data.copy())
    correct = pd.DataFrame(
        {
            "TESTGRP_value": [0, -1, 0, 0, -1, -1, -1, -1, -1],
            "TESTPOINTS_value": [10, -1, 10, 10, -1, -1, -1, -1, -1],
            "TESTDESC_value": [
                "positive",
                None,
                "positive",
                "positive",
                None,
                None,
                None,
                None,
                None,
            ],
        }
    )
    assert_frame_equal(correct, res, check_index_type=False, check_dtype=False)


def test_numerical_range_scorecard_with_default(pd_scorecard_data: "pd.DataFrame"):

    test_scorecard = get_test_scorecard().add_criteria(
        scorecard.ScoreCriteria("value", "numerical")
        .set_other_score(10, "p")
        .add_range_score("-inf", 0, 50, "np")
    )
    res = test_scorecard.score_numerical_vars(pd_scorecard_data.copy())
    correct = pd.DataFrame(
        {
            # "value": pd_scorecard_data["value"],
            # "category": pd_scorecard_data["category"],
            # "category_idx": pd_scorecard_data["category_idx"],
            "TESTGRP_value": [0, 1, 0, 0, 1, 0, 0, 1, 0],
            "TESTPOINTS_value": [10, 50, 10, 10, 50, 10, 10, 50, 10],
            "TESTDESC_value": ["p", "np", "p", "p", "np", "p", "p", "np", "p"],
        }
    )
    assert_frame_equal(correct, res, check_index_type=False, check_dtype=False)


def test_numerical_discrete_score(pd_scorecard_data: "pd.DataFrame"):
    test_scorecard = get_test_scorecard().add_criteria(
        scorecard.ScoreCriteria("value", "numerical").add_discrete_score(
            [1.0, -1.0, 0.0, "inf", None, "nan"], 10, "ints"
        )
    )
    res = test_scorecard.score_numerical_vars(pd_scorecard_data.copy())
    correct = pd.DataFrame(
        {
            # "value": pd_scorecard_data["value"],
            # "category": pd_scorecard_data["category"],
            # "category_idx": pd_scorecard_data["category_idx"],
            "TESTGRP_value": [0, 0, 0, -1, -1, 0, 0, -1, 0],
            "TESTPOINTS_value": [10, 10, 10, -1, -1, 10, 10, -1, 10],
            "TESTDESC_value": [
                "ints",
                "ints",
                "ints",
                None,
                None,
                "ints",
                "ints",
                None,
                "ints",
            ],
        }
    )
    assert_frame_equal(correct, res, check_index_type=False, check_dtype=False)


def test_numerical_discrete_score_with_default(pd_scorecard_data: "pd.DataFrame"):
    test_scorecard = get_test_scorecard().add_criteria(
        scorecard.ScoreCriteria("value", "numerical")
        .add_discrete_score([0.1, -0.1, "-inf"], 10, "dec")
        .set_other_score(50, "ints")
    )
    res = test_scorecard.score_numerical_vars(pd_scorecard_data.copy())
    correct = pd.DataFrame(
        {
            # "value": pd_scorecard_data["value"],
            # "category": pd_scorecard_data["category"],
            # "category_idx": pd_scorecard_data["category_idx"],
            "TESTGRP_value": [1, 1, 1, 0, 0, 1, 1, 0, 1],
            "TESTPOINTS_value": [50, 50, 50, 10, 10, 50, 50, 10, 50],
            "TESTDESC_value": [
                "ints",
                "ints",
                "ints",
                "dec",
                "dec",
                "ints",
                "ints",
                "dec",
                "ints",
            ],
        }
    )
    assert_frame_equal(correct, res, check_index_type=False, check_dtype=False)


def test_numerical_combination(pd_scorecard_data: "pd.DataFrame"):
    # NOTE: Here the implementation details differ between pandas and spark as none is converted to nan
    test_scorecard = get_test_scorecard().add_criteria(
        scorecard.ScoreCriteria("value", "numerical")
        .add_range_score(1.0, "inf", 10, ">=1")  # 0
        .add_discrete_score(["-inf"], 20, "-inf")  # 1
        .add_range_score(-0.1, 0, 30, "[-0.1, 0)")  # 2
        .add_discrete_score([None], 40, "None")  # 3
        .add_discrete_score(["nan"], 50, "nan")  # 4
        .set_other_score(60, "default")  # 5
        .add_discrete_score([], 999, "invalid")  # 6
    )
    res = test_scorecard.score_numerical_vars(pd_scorecard_data.copy())
    correct = pd.DataFrame(
        {
            # "value": pd_scorecard_data["value"],
            # "category": pd_scorecard_data["category"],
            # "category_idx": pd_scorecard_data["category_idx"],
            "TESTGRP_value": [0, 5, 5, 5, 2, 5, 4, 1, 4],
            "TESTPOINTS_value": [10, 60, 60, 60, 30, 60, 50, 20, 50],
            "TESTDESC_value": [
                ">=1",
                "default",
                "default",
                "default",
                "[-0.1, 0)",
                "default",
                "nan",
                "-inf",
                "nan",
            ],
        }
    )

    assert_frame_equal(correct, res, check_index_type=False, check_dtype=False)


def test_numerical_overlapping_ranges(pd_scorecard_data: "pd.DataFrame"):
    test_scorecard = get_test_scorecard().add_criteria(
        scorecard.ScoreCriteria("value", "numerical")
        .add_range_score(0.1, 1.0, 99, "[0.1, 1.0)")  # 0
        .add_range_score(0.001, "inf", 10, ">=0")  # 1
        .add_discrete_score(["-inf"], 20, "-inf")  # 2
        .add_range_score(-0.1, 0, 30, "[-0.1, 0)")  # 3
        .add_discrete_score(["nan"], 50, "nan")  # 4
        .add_discrete_score([None], 40, "None")  # 5
        .set_other_score(60, "default")  # 6
        .add_discrete_score([0], 70, "0")  # 7
    )
    res = test_scorecard.score_numerical_vars(pd_scorecard_data.copy())
    correct = pd.DataFrame(
        {
            # "value": pd_scorecard_data["value"],
            # "category": pd_scorecard_data["category"],
            # "category_idx": pd_scorecard_data["category_idx"],
            "TESTGRP_value": [1, 6, 7, 1, 3, 6, 5, 2, 5],
            "TESTPOINTS_value": [10, 60, 70, 10, 30, 60, 40, 20, 40],
            "TESTDESC_value": [
                ">=0",
                "default",
                "0",
                ">=0",
                "[-0.1, 0)",
                "default",
                "None",
                "-inf",
                "None",
            ],
        }
    )

    assert_frame_equal(correct, res, check_index_type=False, check_dtype=False)


def test_discrete_override_range(pd_scorecard_data: "pd.DataFrame"):
    """Discrete values should always override range values"""
    test_scorecard = get_test_scorecard().add_criteria(
        scorecard.ScoreCriteria("value", "numerical")
        .add_discrete_score([0, 1], 0, "{0, 1}")  # 0
        .add_range_score(0, "inf", 10, ">=0")  # 1
        .add_discrete_score(["-inf"], 20, "-inf")  # 2
        .add_range_score(-0.1, 0, 30, "[-0.1, 0)")  # 3
        .add_discrete_score([None], 40, "None")  # 4
        .add_discrete_score(["nan"], 50, "nan")  # 5
        .set_other_score(60, "default")  # 6
    )
    res = test_scorecard.score_numerical_vars(pd_scorecard_data.copy())
    correct = pd.DataFrame(
        {
            # "value": pd_scorecard_data["value"],
            # "category": pd_scorecard_data["category"],
            # "category_idx": pd_scorecard_data["category_idx"],
            "TESTGRP_value": [0, 6, 0, 1, 3, 6, 5, 2, 5],
            "TESTPOINTS_value": [0, 60, 0, 10, 30, 60, 50, 20, 50],
            "TESTDESC_value": [
                "{0, 1}",
                "default",
                "{0, 1}",
                ">=0",
                "[-0.1, 0)",
                "default",
                "nan",
                "-inf",
                "nan",
            ],
        }
    )

    assert_frame_equal(correct, res, check_index_type=False, check_dtype=False)


def test_discrete_categorical_data(pd_scorecard_data: "pd.DataFrame"):
    test_scorecard = get_test_scorecard().add_criteria(
        scorecard.ScoreCriteria("category", "categorical").add_discrete_score(
            ["a", "b", "c"], 10, "{a,b,c}"
        )
    )
    res = test_scorecard.score_categorical_vars(pd_scorecard_data.copy())
    correct = pd.DataFrame(
        {
            # "value": pd_scorecard_data["value"],
            # "category": pd_scorecard_data["category"],
            # "category_idx": pd_scorecard_data["category_idx"],
            "TESTGRP_category": [0, 0, 0, -1, -1, -1, -1, -1, -1],
            "TESTPOINTS_category": [10, 10, 10, -1, -1, -1, -1, -1, -1],
            "TESTDESC_category": [
                "{a,b,c}",
                "{a,b,c}",
                "{a,b,c}",
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        }
    )

    assert_frame_equal(correct, res, check_index_type=False, check_dtype=False)


def test_discrete_categorical_data_edge_cases(pd_scorecard_data: "pd.DataFrame"):
    test_scorecard = get_test_scorecard().add_criteria(
        scorecard.ScoreCriteria("category", "categorical")
        .add_discrete_score(["a" * 100_000, None, ""], 10, "edge")
        .add_discrete_score(
            [], 99999, "notvalid"
        )  # NOTE: If not careful this could match with ''
    )
    res = test_scorecard.score_categorical_vars(pd_scorecard_data.copy())
    correct = pd.DataFrame(
        {
            # "value": pd_scorecard_data["value"],
            # "category": pd_scorecard_data["category"],
            # "category_idx": pd_scorecard_data["category_idx"],
            "TESTGRP_category": [-1, -1, -1, -1, -1, -1, 0, 0, 0],
            "TESTPOINTS_category": [-1, -1, -1, -1, -1, -1, 10, 10, 10],
            "TESTDESC_category": [
                None,
                None,
                None,
                None,
                None,
                None,
                "edge",
                "edge",
                "edge",
            ],
        }
    )

    assert_frame_equal(correct, res, check_index_type=False, check_dtype=False)


def test_discrete_categorical_data_with_default(pd_scorecard_data: "pd.DataFrame"):
    test_scorecard = get_test_scorecard().add_criteria(
        scorecard.ScoreCriteria("category", "categorical")
        .add_discrete_score(["a" * 100_000, None, ""], 10, "edge")
        .set_other_score(20, "not")
    )

    res = test_scorecard.score_categorical_vars(pd_scorecard_data.copy())
    correct = pd.DataFrame(
        {
            # "value": pd_scorecard_data["value"],
            # "category": pd_scorecard_data["category"],
            # "category_idx": pd_scorecard_data["category_idx"],
            "TESTGRP_category": [1, 1, 1, 1, 1, 1, 0, 0, 0],
            "TESTPOINTS_category": [20, 20, 20, 20, 20, 20, 10, 10, 10],
            "TESTDESC_category": [
                "not",
                "not",
                "not",
                "not",
                "not",
                "not",
                "edge",
                "edge",
                "edge",
            ],
        }
    )

    assert_frame_equal(correct, res, check_index_type=False, check_dtype=False)


def test_scorecard_integration(
    pd_scorecard_data: "pd.DataFrame", pd_scorecard_result_data: "pd.DataFrame"
):
    test_scorecard = (
        get_test_scorecard(
            score_scaling_params={
                "base_points": 100,
                "base_odds": 5,
                "pdo": 9,
            }
        )
        .add_criteria(
            scorecard.ScoreCriteria("value", "numerical")
            .add_range_score("inf", "inf", 99, "No Value")
            .add_range_score(0.001, "inf", 10, ">=0")
            .add_discrete_score(["-inf"], 20, "-inf")
            .add_range_score(-0.1, 0, 30, "[-0.1, 0)")
            .add_discrete_score([None], 40, "None")
            .add_discrete_score(["nan"], 50, "nan")
            .set_other_score(60, "default")
            .add_discrete_score([0], 70, "0")
        )
        .add_criteria(
            scorecard.ScoreCriteria("category", "categorical")
            .set_other_score(80, "default")
            .add_discrete_score(["a"], 90, "a")
            .add_discrete_score(["regex:[b-c]"], 100, "b,c")
            .add_discrete_score(["d", "e", "f", None], 110, "d,e,f,None")
        )
        .add_criteria(
            scorecard.ScoreCriteria("category_idx", "numerical")
            .add_discrete_score([1], 120, "1")
            .add_discrete_score([2, 3], 130, "2,3")
            .add_discrete_score([], 9_999_999, "emptyset")
            .add_discrete_score([None], 140, "None")
        )
    )
    res = test_scorecard.score_df(pd_scorecard_data.copy())
    correct_res_pd = pd_scorecard_result_data

    assert_frame_equal(
        correct_res_pd.drop(
            columns=["TESTPOINTS_SUM_PD_LOGODDS", "TESTPOINTS_SUM_PD"], inplace=False
        ).sort_index(axis=1),
        res.sort_index(axis=1),
        check_index_type=False,
        check_dtype=False,
        check_names=True,
    )
    res = test_scorecard.score_pd(res)
    assert_frame_equal(
        correct_res_pd.sort_index(axis=1),
        res.sort_index(axis=1),
        check_index_type=False,
        check_dtype=False,
        check_names=True,
    )


def test_scorecard_integration_using_stored_values(
    pd_scorecard_data: "pd.DataFrame", pd_scorecard_result_data: "pd.DataFrame"
):
    test_scorecard = (
        get_test_scorecard(
            score_scaling_params={
                "base_points": 100,
                "base_odds": 5,
                "pdo": 9,
            }
        )
        .add_criteria(
            scorecard.ScoreCriteria(pd_scorecard_data["value"], "numerical")
            .add_range_score("inf", "inf", 99, "No Value")
            .add_range_score(0.001, "inf", 10, ">=0")
            .add_discrete_score(["-inf"], 20, "-inf")
            .add_range_score(-0.1, 0, 30, "[-0.1, 0)")
            .add_discrete_score([None], 40, "None")
            .add_discrete_score(["nan"], 50, "nan")
            .set_other_score(60, "default")
            .add_discrete_score([0], 70, "0")
        )
        .add_criteria(
            scorecard.ScoreCriteria(pd_scorecard_data["category"], "categorical")
            .set_other_score(80, "default")
            .add_discrete_score(["a"], 90, "a")
            .add_discrete_score(["regex:[b-c]"], 100, "b,c")
            .add_discrete_score(["d", "e", "f", None], 110, "d,e,f,None")
        )
        .add_criteria(
            scorecard.ScoreCriteria(pd_scorecard_data["category_idx"], "numerical")
            .add_discrete_score([1], 120, "1")
            .add_discrete_score([2, 3], 130, "2,3")
            .add_discrete_score([], 9_999_999, "emptyset")
            .add_discrete_score([None], 140, "None")
        )
    )
    res = test_scorecard.execute(
        pd_scorecard_data, final_vars=["ScoreCardModel.score_pd"]
    )
    correct_res_pd = pd_scorecard_result_data
    assert_frame_equal(
        correct_res_pd.sort_index(axis=1),
        res.sort_index(axis=1),
        check_index_type=False,
        check_dtype=False,
        check_names=True,
    )
    res = test_scorecard.execute(pd_scorecard_data)

    assert_frame_equal(
        correct_res_pd.drop(
            columns=["TESTPOINTS_SUM_PD_LOGODDS", "TESTPOINTS_SUM_PD"], inplace=False
        ).sort_index(axis=1),
        res.sort_index(axis=1),
        check_index_type=False,
        check_dtype=False,
        check_names=True,
    )


def test_adjustment_model_no_variables(
    pd_scorecard_result_data: "pd.DataFrame",
    adjustment_scorecard_model: "scorecard.Score",
):
    # Test using non adjusted
    res = adjustment_scorecard_model.adjust_scores(
        pd_scorecard_result_data.copy(),
        adjustment_model={
            "score_prefix": "ADJUSTED_TEST_",
            "variable_score_adjustments": [],
        },
        include_non_adjusted_scores=True,
        include_pd=False,
    )
    gt_result_df = pd_scorecard_result_data.copy()
    gt_result_df["ADJUSTED_TEST_SUM"] = gt_result_df["TESTPOINTS_SUM"]
    assert_frame_equal(gt_result_df, res, check_index_type=False, check_dtype=False)
    # Test only using adjusted
    # TODO maybe add better validation
    # with pytest.raises(TypeError):
    # NOTE: The spark implementation raises a error here when there are no columns to sum
    # This implementation returns a 0 for the sum. I feel not raising an error might be a better approach
    res = adjustment_scorecard_model.adjust_scores(
        pd_scorecard_result_data.copy(),
        adjustment_model={
            "score_prefix": "ADJUSTED_TEST_",
            "variable_score_adjustments": [],
        },
        include_non_adjusted_scores=False,
        include_pd=False,
    )
    assert np.allclose(res["ADJUSTED_TEST_SUM"], 0)


def test_adjustment_model_none(
    pd_scorecard_result_data: "pd.DataFrame",
    adjustment_scorecard_model: "scorecard.Score",
):
    # NOTE: Might be better to break this up into 3 tests
    # Test single operation no other scores
    res = adjustment_scorecard_model.adjust_scores(
        pd_scorecard_result_data.copy(),
        adjustment_model={
            "score_prefix": "ADJUSTED_TEST_",
            "variable_score_adjustments": [
                {"operation": None, "variable": "TESTPOINTS_value"}
            ],
        },
        include_non_adjusted_scores=False,
        include_pd=False,
    )
    gt_result_df = pd_scorecard_result_data.copy()
    gt_result_df["ADJUSTED_TEST_SUM"] = gt_result_df["TESTPOINTS_value"]
    assert_frame_equal(gt_result_df, res, check_index_type=False, check_dtype=False)
    # Test multiple operation no other scores
    res = adjustment_scorecard_model.adjust_scores(
        pd_scorecard_result_data.copy(),
        adjustment_model={
            "score_prefix": "ADJUSTED_TEST_",
            "variable_score_adjustments": [
                {"operation": "", "variable": "TESTPOINTS_category"},
                {"operation": "none", "variable": "TESTPOINTS_category_idx"},
            ],
        },
        include_non_adjusted_scores=False,
        include_pd=False,
    )
    gt_result_df["ADJUSTED_TEST_SUM"] = (
        gt_result_df["TESTPOINTS_category"] + gt_result_df["TESTPOINTS_category_idx"]
    )
    assert_frame_equal(gt_result_df, res, check_index_type=False, check_dtype=False)
    # Test include pd and non_adjusted
    res = adjustment_scorecard_model.adjust_scores(
        pd_scorecard_result_data.copy(),
        adjustment_model={
            "score_prefix": "ADJUSTED_TEST_",
            "variable_score_adjustments": [
                {"operation": "None", "variable": "TESTPOINTS_category"}
            ],
        },
        include_non_adjusted_scores=True,
        include_pd=True,
    )
    gt_result_df["ADJUSTED_TEST_SUM"] = gt_result_df["TESTPOINTS_SUM"]
    gt_result_df["ADJUSTED_TEST_SUM_PD_LOGODDS"] = gt_result_df[
        "TESTPOINTS_SUM_PD_LOGODDS"
    ]
    gt_result_df["ADJUSTED_TEST_SUM_PD"] = gt_result_df["TESTPOINTS_SUM_PD"]
    assert_frame_equal(gt_result_df, res, check_index_type=False, check_dtype=False)


def test_adjustment_model_constant(
    pd_scorecard_data: "pd.DataFrame",
    pd_scorecard_result_data: "pd.DataFrame",
    adjustment_scorecard_model: "scorecard.Score",
):
    # Test Integer value
    res = adjustment_scorecard_model.adjust_scores(
        pd_scorecard_result_data.copy(),
        adjustment_model={
            "score_prefix": "ADJUSTED_TEST_",
            "variable_score_adjustments": [
                {
                    "operation": "constant",
                    "variable": "TESTPOINTS_category",
                    "value": 100,
                }
            ],
        },
        include_non_adjusted_scores=False,
        include_pd=False,
    )
    gt_result_df = pd_scorecard_result_data.copy()
    gt_result_df["ADJUSTED_TEST_SUM"] = 100
    assert_frame_equal(gt_result_df, res, check_index_type=False, check_dtype=False)
    # Test float value
    res = adjustment_scorecard_model.adjust_scores(
        pd_scorecard_result_data.copy(),
        adjustment_model={
            "score_prefix": "ADJUSTED_TEST_",
            "variable_score_adjustments": [
                {
                    "operation": "constant",
                    "variable": "TESTPOINTS_category",
                    "value": 3.14,
                }
            ],
        },
        include_non_adjusted_scores=True,
        include_pd=False,
    )
    gt_result_df = pd_scorecard_result_data.copy()
    gt_result_df["ADJUSTED_TEST_SUM"] = (
        gt_result_df["TESTPOINTS_SUM"] - gt_result_df["TESTPOINTS_category"] + 3.14
    )
    assert_frame_equal(gt_result_df, res, check_index_type=False, check_dtype=False)
    # Test string value
    res = adjustment_scorecard_model.adjust_scores(
        pd.concat([pd_scorecard_result_data, pd_scorecard_data], axis=1),
        adjustment_model={
            "score_prefix": "ADJUSTED_TEST_",
            "variable_score_adjustments": [
                {"operation": "constant", "variable": "category", "value": "3.14"}
            ],
        },
        include_non_adjusted_scores=True,
        include_pd=False,
    )
    gt_result_df = pd.concat([pd_scorecard_result_data, pd_scorecard_data], axis=1)
    gt_result_df["ADJUSTED_TEST_SUM"] = gt_result_df["TESTPOINTS_SUM"] + 3.14
    assert_frame_equal(gt_result_df, res, check_index_type=False, check_dtype=False)
    # Test invalid value
    with pytest.raises(ValueError):
        res = adjustment_scorecard_model.adjust_scores(
            pd.concat([pd_scorecard_result_data, pd_scorecard_data], axis=1),
            adjustment_model={
                "score_prefix": "ADJUSTED_TEST_",
                "variable_score_adjustments": [
                    {
                        "operation": "constant",
                        "variable": "category",
                        "value": "not a valid value",
                    }
                ],
            },
            include_non_adjusted_scores=True,
            include_pd=False,
        )


def test_adjustment_model_sum(
    pd_scorecard_result_data: "pd.DataFrame",
    adjustment_scorecard_model: "scorecard.Score",
):
    # Test non adjusted off
    res = adjustment_scorecard_model.adjust_scores(
        pd_scorecard_result_data.copy(),
        adjustment_model={
            "score_prefix": "ADJUSTED_TEST_",
            "variable_score_adjustments": [
                {"operation": "Sum", "variable": "TESTPOINTS_category", "value": 3.14}
            ],
        },
        include_non_adjusted_scores=False,
        include_pd=False,
    )
    gt_result_df = pd_scorecard_result_data.copy()
    gt_result_df["ADJUSTED_TEST_SUM"] = gt_result_df["TESTPOINTS_category"] + 3.14
    assert_frame_equal(gt_result_df, res, check_index_type=False, check_dtype=False)
    # Test non adjusted on
    res = adjustment_scorecard_model.adjust_scores(
        pd_scorecard_result_data.copy(),
        adjustment_model={
            "score_prefix": "ADJUSTED_TEST_",
            "variable_score_adjustments": [
                {"operation": "sum", "variable": "TESTPOINTS_category", "value": -10}
            ],
        },
        include_non_adjusted_scores=True,
        include_pd=False,
    )
    gt_result_df = pd_scorecard_result_data.copy()
    gt_result_df["ADJUSTED_TEST_SUM"] = gt_result_df["TESTPOINTS_SUM"] - 10
    assert_frame_equal(gt_result_df, res, check_index_type=False, check_dtype=False)


def test_adjust_scores_invalid_op(
    pd_scorecard_result_data: "pd.DataFrame",
    adjustment_scorecard_model: "scorecard.Score",
):
    # Test non adjusted off
    with pytest.raises(ValueError):
        res = adjustment_scorecard_model.adjust_scores(
            pd_scorecard_result_data.copy(),
            adjustment_model={
                "score_prefix": "ADJUSTED_TEST_",
                "variable_score_adjustments": [
                    {
                        "operation": "not_valid",
                        "variable": "TESTPOINTS_category",
                        "value": 3.14,
                    }
                ],
            },
            include_non_adjusted_scores=False,
            include_pd=False,
        )
    with pytest.raises(ValueError):
        res = adjustment_scorecard_model.adjust_scores(
            pd_scorecard_result_data.copy(),
            adjustment_model={
                "score_prefix": "ADJUSTED_TEST_",
                "variable_score_adjustments": [
                    {
                        "operation": "sum",
                        "variable": "column_does_not_exist",
                        "value": 3.14,
                    }
                ],
            },
            include_non_adjusted_scores=False,
            include_pd=False,
        )


# def test_scorecard_load_and_save():

#     test_scorecard = get_test_scorecard(
#         score_scaling_params={
#             "base_points": 100,
#             "base_odds": 5,
#             "pdo": 9,
#         }
#     )\
#         .add_criteria(
#             scorecard.ScoreCriteria("value", "numerical")
#                 .add_range_score("inf", "inf", 99, "No Value")
#                 .add_range_score(0.001, "inf", 10, ">=0")
#                 .add_discrete_score(["-inf"], 20, "-inf")
#                 .add_range_score(-0.1, 0, 30, "[-0.1, 0)")
#                 .add_discrete_score([None], 40, "None")
#                 .add_discrete_score(["nan"], 50, "nan")
#                 .set_other_score(60, "default")
#                 .add_discrete_score([0], 70, "0")
#         )\
#         .add_criteria(
#             scorecard.ScoreCriteria("category", "categorical")
#                 .set_other_score(80, "default")
#                 .add_discrete_score(["a"], 90, "a")
#                 .add_discrete_score(["regex:[b-c]"], 100, "b,c")
#                 .add_discrete_score(["d", "e", "f", None], 110, "d,e,f,None")
#         )\
#         .add_criteria(
#             scorecard.ScoreCriteria("category_idx", "numerical")
#                 .add_discrete_score([1], 120, "1")
#                 .add_discrete_score([2, 3], 130, "2,3")
#                 .add_discrete_score([], 9_999_999, "emptyset")
#                 .add_discrete_score([None], 140, "None")
#         )

#     with patch("builtins.open", mock_open(read_data="data")) as mock_file:
#         test_scorecard.save("test.json")
#     with patch("builtins.open", mock_open(read_data="data")) as mock_file:
#         test_scorecard.save("test.yaml")
#     with patch("builtins.open", mock_open(read_data="data")) as mock_file:
#         test_scorecard.save("test.json")

# json_data = adjustment_scorecard_model.model_dump_json()
# with patch("builtins.open", mock_open(read_data=json_data)) as mock_file:
#     adjustment_scorecard_model.from_config("test.json")
# with patch("builtins.open", mock_open(read_data=json_data)) as mock_file:
#     adjustment_scorecard_model.from_config("test.yaml")


def test_invalid_scorecards():
    with pytest.raises(
        ValueError,
        match=r"Invalid Criteria type numical expecting either \"numerical\" or \"categorical\"",
    ):
        test_scorecard = get_test_scorecard().add_criteria(
            scorecard.ScoreCriteria("value", "numical")
        )

    with pytest.raises(TypeError, match=r"missing 1 required positional argument"):
        test_scorecard = get_test_scorecard().add_criteria(
            scorecard.ScoreCriteria("value", "numerical").add_range_score(0, 10, "desc")
        )
    # test_model = get_test_scorecard(
    #     variable_params=[
    #         (
    #             scorecard_model.ScoreCriteria("value", "numerical").add_range_score(
    #                 [0, 10], "asdf", "desc"
    #             )
    #         ),
    #     ],
    # )
    # with pytest.raises(
    #     scorecard.ScoreCardValidationError, match=r"Input should be a valid number"
    # ):
    #     test_scorecard = scorecard.Score(model=test_model.as_dict(), logger=logger)
    # test_model = get_test_scorecard(
    #     variable_params=[
    #         (
    #             scorecard_model.ScoreCriteria("value", "numerical").add_discrete_score(
    #                 [], "asdf", "desc"
    #             )
    #         ),
    #     ],
    # )
    # with pytest.raises(
    #     scorecard.ScoreCardValidationError, match=r"Input should be a valid number"
    # ):
    #     test_scorecard = scorecard.Score(model=test_model.as_dict(), logger=logger)
    # test_model = get_test_scorecard(
    #     variable_params=[
    #         (
    #             scorecard_model.ScoreCriteria(
    #                 "value", "categorical"
    #             ).add_discrete_score([], 10, "desc", 1.1)
    #         ),
    #     ],
    # )
    # with pytest.raises(
    #     scorecard.ScoreCardValidationError, match=r"Input should be a valid integer"
    # ) as err:
    #     test_scorecard = scorecard.Score(model=test_model.as_dict(), logger=logger)
