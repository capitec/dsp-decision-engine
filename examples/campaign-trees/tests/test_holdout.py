from campaign_trees.holdout import is_control, is_universal_holdout

CLIENTS = [str(i) for i in range(2000)]


def test_a_client_stays_in_the_same_group_every_time_they_are_asked():
    assert [is_control(c, 23, "hv1", 0.05) for c in CLIENTS] == [is_control(c, 23, "hv1", 0.05) for c in CLIENTS]


def test_the_control_group_is_about_the_size_asked_for():
    assert 60 <= sum(is_control(c, 23, "hv1", 0.05) for c in CLIENTS) <= 140


def test_raising_the_percentage_only_adds_clients_to_the_group():
    five = {c for c in CLIENTS if is_control(c, 23, "hv1", 0.05)}
    ten = {c for c in CLIENTS if is_control(c, 23, "hv1", 0.10)}
    assert five < ten


def test_a_new_design_version_draws_a_different_group():
    assert {c for c in CLIENTS if is_control(c, 23, "hv1", 0.05)} != {c for c in CLIENTS if is_control(c, 23, "hv2", 0.05)}


def test_each_campaign_draws_its_own_control_group():
    assert {c for c in CLIENTS if is_control(c, 23, "hv1", 0.05)} != {c for c in CLIENTS if is_control(c, 24, "hv1", 0.05)}


def test_the_universal_holdout_does_not_depend_on_the_campaign():
    assert 5 <= sum(is_universal_holdout(c, "uhv1", 0.01) for c in CLIENTS) <= 40
