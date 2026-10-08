from limit_mgmt import decrease
from limit_mgmt.vocab import NoticeClass


def test_emerging_arrears_fires_the_decrease_trigger():
    assert decrease.decrease_trigger_fires(1) is True and decrease.decrease_trigger_fires(0) is False


def test_the_new_limit_covers_the_balance_authorisations_and_accrued_interest():
    assert decrease.decrease_target_limit(True, 15_000.0, 8_200.0, 150.0, 95.0) == 8_445.0


def test_a_decrease_never_raises_the_limit():
    assert decrease.decrease_target_limit(True, 5_000.0, 8_000.0, 0.0, 0.0) == 5_000.0


def test_no_trigger_leaves_the_limit_alone():
    assert decrease.decrease_target_limit(False, 15_000.0, 8_200.0, 150.0, 95.0) == 15_000.0


def test_emerging_arrears_is_immediate_class():
    assert decrease.decrease_notice_class_code(True) == NoticeClass.IMMEDIATE
    assert decrease.decrease_notice_class_code(False) == NoticeClass.NOT_APPLICABLE
