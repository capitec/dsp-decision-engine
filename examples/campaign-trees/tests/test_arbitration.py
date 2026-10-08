from campaign_trees.arbitration import Qualification, arbitrate

SMS, IN_APP = 2, 1


def q(client, campaign, value, channels=(SMS, IN_APP), **kw):
    return Qualification(client, campaign, 1.0, value, channels, **kw)


def contacts(results):
    return {(r.client_id, r.campaign_id): (r.channel, r.reason) for r in results}


def test_the_campaign_with_the_higher_value_wins_a_clients_last_slot():
    results = arbitrate([q("c1", 1, 10.0), q("c1", 2, 30.0), q("c1", 3, 20.0)], {SMS: 9, IN_APP: 9})
    assert contacts(results) == {("c1", 2): (SMS, None), ("c1", 3): (SMS, None), ("c1", 1): (None, "lost_on_rank:3")}


def test_a_full_channel_falls_through_to_the_next_preference_then_to_no_contact():
    results = arbitrate([q("c1", 1, 30.0), q("c2", 1, 20.0), q("c3", 1, 10.0)], {SMS: 1, IN_APP: 1})
    assert contacts(results) == {("c1", 1): (SMS, None), ("c2", 1): (IN_APP, None),
                                 ("c3", 1): (None, "channel_capacity_exhausted")}


def test_control_and_suspended_campaigns_are_never_contacted_and_say_why():
    results = arbitrate([q("c1", 1, 5.0, is_control=True), q("c2", 1, 5.0, campaign_suspended=True)], {SMS: 9})
    assert contacts(results) == {("c1", 1): (None, "control_group"), ("c2", 1): (None, "campaign_suspended")}


def test_a_control_client_does_not_use_up_capacity_or_a_contact_slot():
    results = arbitrate([q("c1", 1, 50.0, is_control=True), q("c1", 2, 10.0)], {SMS: 1})
    assert contacts(results)[("c1", 2)] == (SMS, None)


def test_the_result_does_not_depend_on_the_order_of_the_input():
    batch = [q(f"c{i % 7}", i % 3, float(i % 5)) for i in range(40)]
    assert arbitrate(batch, {SMS: 12, IN_APP: 8}) == arbitrate(batch[::-1], {SMS: 12, IN_APP: 8})


def test_no_channel_is_used_beyond_its_capacity():
    batch = [q(f"c{i}", 1, float(i)) for i in range(50)]
    used = [r.channel for r in arbitrate(batch, {SMS: 10, IN_APP: 5}) if r.channel]
    assert used.count(SMS) == 10 and used.count(IN_APP) == 5
