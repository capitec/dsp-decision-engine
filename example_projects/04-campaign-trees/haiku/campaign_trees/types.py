"""Types for campaign targeting trees."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import date


@dataclass
class TreeLeaf:
    """A decision leaf in a campaign tree."""
    key: str
    outcome_code: int  # 1=target, 2=control, 3=do_not_target
    tier_code: int  # 1-7
    amount_rule: str  # Fixed amount or expression
    channels: List[str]  # Preferred channels
    priority_weight: float  # 0..1
    reason_label: int  # ~220 possible reasons
    reason_text: str = ""


@dataclass
class TreeNode:
    """A node in a campaign tree."""
    key: str
    level: int
    condition: str  # Readable condition
    features: List[str]  # Features used


@dataclass
class TreeEvaluationResult:
    """Result of evaluating a tree on a client."""
    client_id: int
    campaign_id: int
    tree_version: int
    leaf: TreeLeaf
    path: List[str]  # List of node keys visited
    path_nodes: List[TreeNode] = field(default_factory=list)
    unadjusted_leaf: Optional[TreeLeaf] = None
    overlay_stack_id: int = 0
    decision_id: str = ""
    cycle_date: Optional[date] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "campaign_id": self.campaign_id,
            "tree_version": self.tree_version,
            "leaf_key": self.leaf.key,
            "leaf_outcome_code": self.leaf.outcome_code,
            "leaf_tier_code": self.leaf.tier_code,
            "amount_rule": self.leaf.amount_rule,
            "channels": self.channels,
            "priority_weight": self.leaf.priority_weight,
            "reason_label": self.leaf.reason_label,
            "reason_text": self.leaf.reason_text,
            "path": self.path,
            "path_length": len(self.path),
            "overlay_stack_id": self.overlay_stack_id,
            "unadjusted_leaf_key": self.unadjusted_leaf.key if self.unadjusted_leaf else None,
            "decision_id": self.decision_id,
        }


@dataclass
class Suppression:
    """A suppression record."""
    client_id: int
    suppression_code: int
    campaign_id: int
    channel_code: Optional[int] = None
    scope: str = "all_campaigns"  # all_campaigns, per_campaign, per_channel
    reason_text: str = ""


@dataclass
class Assignment:
    """An assignment of a client to a campaign."""
    assignment_id: str
    client_id: int
    campaign_id: int
    cycle_id: int
    cycle_date: date
    tree_version: int
    variant: int = 1  # 1=champion, 2=challenger1, 3=challenger2
    is_control: bool = False
    leaf_key: str = ""
    offer_tier_code: int = 1
    offered_amount: float = 0.0
    term_months: int = 60
    channel: str = "sms"  # Primary channel
    priority_weight: float = 0.5
    reason_label: int = 0
    reason_text: str = ""
    contacted: bool = True
    not_contacted_reason: str = ""  # Why not contacted if applicable


@dataclass
class ArbitrationResult:
    """Result of arbitration for a client."""
    client_id: int
    cycle_id: int
    assignments: List[Assignment] = field(default_factory=list)
    channel_capacity_used: Dict[str, int] = field(default_factory=dict)
    fatigue_cap_reached: bool = False
