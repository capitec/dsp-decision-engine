"""Tests for affordability capability."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.affordability import Affordability, AffordabilityParams


class TestAffordabilityModule:
    """Test the affordability assessment logic."""

    def test_living_expenses_applies_floor(self):
        """Living expenses should not go below the statutory floor."""
        params = AffordabilityParams()
        result = Affordability.score(
            {
                "declared_living_expenses": 1000.0,  # Below floor
                "gross_monthly_income": 50000.0,
                "dependants_count": 0,
                "num_accounts": 0,
                "average_account_instalment": 0.0,
                "instalment": 5000.0,
            },
            params={"affordability": vars(params)},
        )
        # Floor = 3000 + (500 * 0) = 3000
        assert result["living_expenses"] == pytest.approx(3000.0)

    def test_living_expenses_respects_declared(self):
        """When declared expenses are above floor, use declared."""
        params = AffordabilityParams()
        result = Affordability.score(
            {
                "declared_living_expenses": 8000.0,  # Above floor
                "gross_monthly_income": 50000.0,
                "dependants_count": 0,
                "num_accounts": 0,
                "average_account_instalment": 0.0,
                "instalment": 5000.0,
            },
            params={"affordability": vars(params)},
        )
        assert result["living_expenses"] == pytest.approx(8000.0)

    def test_discretionary_income_calculation(self):
        """Discretionary income = net income - expenses - obligations."""
        params = AffordabilityParams()
        result = Affordability.score(
            {
                "declared_living_expenses": 5000.0,
                "gross_monthly_income": 45000.0,  # Not used directly
                "dependants_count": 0,
                "num_accounts": 3,
                "average_account_instalment": 1000.0,
                "instalment": 5000.0,
                "net_monthly_income": 40000.0,  # 45000 * 0.89 roughly
            },
            params={"affordability": vars(params)},
        )
        # Discretionary = 40000 - 5000 - 3000 = 32000
        assert result["discretionary_income"] == pytest.approx(32000.0)

    def test_affordability_pass_verdict(self):
        """Verdict should be 1 (pass) when there's sufficient discretionary income."""
        params = AffordabilityParams()
        result = Affordability.score(
            {
                "declared_living_expenses": 5000.0,
                "gross_monthly_income": 50000.0,
                "dependants_count": 1,
                "num_accounts": 2,
                "average_account_instalment": 500.0,
                "instalment": 5000.0,
                "net_monthly_income": 41000.0,  # sufficient
            },
            params={"affordability": vars(params)},
        )
        # Living expenses floor = 3000 + 500 = 3500
        # Obligations = 2 * 500 = 1000
        # Discretionary = 41000 - 3500 - 1000 = 36500
        # Check against instalment: 36500 > 5000 * threshold(0.5) → pass
        assert result["affordability_verdict_code"] == 1

    def test_max_affordable_instalment(self):
        """Max affordable instalment applies buffer to discretionary income."""
        params = AffordabilityParams(affordability_buffer=0.35)
        result = Affordability.score(
            {
                "declared_living_expenses": 5000.0,
                "gross_monthly_income": 50000.0,
                "dependants_count": 0,
                "num_accounts": 0,
                "average_account_instalment": 0.0,
                "instalment": 5000.0,
                "net_monthly_income": 41000.0,
            },
            params={"affordability": vars(params)},
        )
        # Discretionary = 41000 - 5000 - 0 = 36000
        # Max affordable = 36000 * 0.35 = 12600
        assert result["max_affordable_instalment"] == pytest.approx(12600.0)

    def test_equivalence_across_modes(self):
        """Test that interpreted, stepped, and fused modes agree."""
        from decider2.testing import assert_equivalent

        params = AffordabilityParams()
        test_case = {
            "declared_living_expenses": 4500.0,
            "gross_monthly_income": 50000.0,
            "dependants_count": 2,
            "num_accounts": 1,
            "average_account_instalment": 800.0,
            "instalment": 4000.0,
            "net_monthly_income": 41000.0,
        }

        assert_equivalent(
            Affordability,
            test_case,
            params={"affordability": vars(params)},
        )
