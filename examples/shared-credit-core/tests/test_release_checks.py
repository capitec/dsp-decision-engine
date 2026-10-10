"""What a release check sees of the quote pipeline: the overlays, reason codes and cut-overs are in the diff."""
import json
import warnings

from decikit import ConfigurableStep, Engine, HiddenStateWarning
from decikit.lab import diff

from credit_core.pipeline import build


def document(path, edit=lambda doc: None):
    doc = json.loads(path.read_text())
    edit(doc)
    return ConfigurableStep.load(doc)


def rebuilt(rate_card, configs, adjustments=lambda doc: None, reasons=lambda doc: None):
    return build(rate_card, document(configs / "score_adjustments.json", adjustments),
                 document(configs / "reason_codes.json", reasons))


def test_the_quote_pipeline_reads_no_hidden_objects(quote_pipeline):
    with warnings.catch_warnings():
        warnings.simplefilter("error", HiddenStateWarning)
        Engine().bind(quote_pipeline, mode="interpreted")


def test_the_same_documents_build_an_unchanged_pipeline(quote_pipeline, rate_card, configs):
    assert diff.structure(quote_pipeline, rebuilt(rate_card, configs)).identical


def test_doubling_a_score_overlay_changes_the_overlay_step(quote_pipeline, rate_card, configs):
    doubled = rebuilt(rate_card, configs, lambda doc: doc["adjustments"][0]["effect"].update(value=-36.0))
    changed = diff.structure(quote_pipeline, doubled).changed_steps
    assert [c.ref.path for c in changed] == ["flex_loan_quote/apply_score_adjustments"] and changed[0].consts


def test_a_governance_only_edit_to_an_overlay_is_not_a_change(quote_pipeline, rate_card, configs):
    def reword(doc):
        doc["adjustments"][0].update(rationale="reworded", owner="Model Risk", review_date="2027-03-31")

    assert diff.structure(quote_pipeline, rebuilt(rate_card, configs, reword)).identical


def test_changing_a_reason_severity_changes_the_ranking_step(quote_pipeline, rate_card, configs):
    reranked = rebuilt(rate_card, configs, reasons=lambda doc: doc["codes"][0].update(severity_rank=99))
    changed = diff.structure(quote_pipeline, reranked).changed_steps
    assert [c.ref.path for c in changed] == ["flex_loan_quote/reasons/rank_reasons"]
