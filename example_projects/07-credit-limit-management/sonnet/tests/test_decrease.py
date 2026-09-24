from limit_mgmt.decrease import decrease_target_limit, decrease_trigger_fires
from limit_mgmt.vocab import NoticeClass
from limit_mgmt.decrease import decrease_notice_class


def test_d03_fires_on_current_arrears():
    assert decrease_trigger_fires(worst_arrears_months_now=1) is True
    assert decrease_trigger_fires(worst_arrears_months_now=0) is False


def test_floor_is_balance_plus_authorisations_plus_interest():
    """§5.7 rule 1: "the new limit must be at least the statement balance plus unsettled
    authorisations plus accrued unbilled interest."""
    target = decrease_target_limit(
        decrease_trigger_fires=True, current_limit=15_000.0,
        statement_balance=8_200.0, unsettled_authorisations=150.0, accrued_unbilled_interest=95.0,
    )
    assert target == 8_445.0


def test_floor_never_exceeds_current_limit():
    target = decrease_target_limit(
        decrease_trigger_fires=True, current_limit=5_000.0,
        statement_balance=8_000.0, unsettled_authorisations=0.0, accrued_unbilled_interest=0.0,
    )
    assert target == 5_000.0  # the floor formula would exceed current_limit; never increases


def test_d03_is_immediate_class():
    assert decrease_notice_class(decrease_trigger_fires=True) == NoticeClass.IMMEDIATE
    assert decrease_notice_class(decrease_trigger_fires=False) == NoticeClass.NOT_APPLICABLE
