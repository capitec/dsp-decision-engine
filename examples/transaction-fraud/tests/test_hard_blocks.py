import pytest
from decikit import FusedEngine, flow

from fraud_interdiction import hard_blocks, vocab
from fraud_interdiction.vocab import Action

GATE_INPUTS = {1: "sanctions_list_match", 2: "account_frozen", 4: "mule_watchlist_confirmed", 8: "device_blocked",
               16: "card_compromised", 32: "court_order_hold"}


@pytest.fixture(scope="module")
def gates():
    executable = FusedEngine(strict_compile=True).bind(flow(hard_blocks.hard_block_step, name="gates"))
    return lambda **held: executable.run(held)


def test_no_gate_forces_nothing(gates):
    out = gates()
    assert (out["hard_block_code"], out["hard_block_forced_action"]) == (0, 0)


@pytest.mark.parametrize("bit,name", GATE_INPUTS.items())
def test_each_gate_sets_its_bit_and_forces_the_action_the_vocabulary_names(gates, bit, name):
    out = gates(**{name: True})
    assert out["hard_block_code"] == bit
    assert out["hard_block_forced_action"] == vocab.GATES[bit][1]


def test_when_two_gates_hold_both_are_recorded_and_the_more_severe_action_applies(gates):
    out = gates(account_frozen=True, sanctions_list_match=True, court_order_hold=True)
    assert out["hard_block_code"] == 1 | 2 | 32
    assert out["hard_block_forced_action"] == Action.BLOCK_CHANNEL


def test_every_gate_has_a_unique_reason_code():
    assert len({code for _, _, code in vocab.GATES.values()}) == len(vocab.GATES)
