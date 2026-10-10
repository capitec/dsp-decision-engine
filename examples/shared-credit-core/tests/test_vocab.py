from credit_core import vocab


def test_the_four_kinds_of_null_stay_distinct():
    assert len({kind.value for kind in vocab.NullKind}) == 4


def test_unavailable_fraud_is_not_an_approval():
    assert vocab.FraudVerdictCode.UNAVAILABLE != vocab.FraudVerdictCode.APPROVE


def test_nested_entities_have_their_own_roles():
    assert vocab.entity_role(2) == "entity_2" and vocab.entity_role(2) != vocab.entity_role(3)


def test_every_product_has_a_name():
    assert vocab.PRODUCT_NAMES[vocab.PRODUCT_FLEX_LOAN] == "Flex Loan"
    assert len(vocab.PRODUCT_NAMES) == 8
