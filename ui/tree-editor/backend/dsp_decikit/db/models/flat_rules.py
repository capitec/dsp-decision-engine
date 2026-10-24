import typing as t
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP
from sqlmodel import SQLModel, Field, Relationship
from .users import User


class FlatRuleProject(SQLModel, table=True):
    __tablename__ = "flat_rule_projects"
    __table_args__ = {"schema": "app_schema"}

    id: t.Optional[uuid.UUID] = Field(
        default_factory=uuid.uuid4, sa_column=Column(UUID, primary_key=True)
    )
    name: str = Field(min_length=1, max_length=255, description="Project name")
    description: t.Optional[str] = Field(
        default=None, max_length=1000, description="Project description"
    )
    created_by: uuid.UUID = Field(
        sa_column=Column(UUID, ForeignKey("app_schema.users.id"))
    )
    project_data: t.Dict[str, t.Any] = Field(
        sa_column=Column(JSONB),
        description="Complete PED prioritized_tree structure with UI metadata",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )

    # Relationships
    created_by_user: User = Relationship()


# Example project_data structure (PrioritizedTreeModule):
# {
#   "type": "prioritized_tree",
#   "input_schema": [["field1", "String"], ["field2", "Int64"]],  // Optional
#   "output": {
#     "data": [
#       {
#         "Action": {"id": 1, "Type": "NCALimitAmount", "Amount": 100, "Recipient": "BANKS"},
#         "number": 10,
#         "Outcome": "Action1"
#       }
#     ],
#     "dtypes": [
#       ["Action", {"type": "Struct", "type_id": "uuid-for-action-type", "fields": {...}}],
#       ["number", "Float64"],
#       ["Outcome", {"type": "String", "type_id": "uuid-for-outcome-categories"}]
#     ],
#     "type_defs": {
#       "uuid-for-action-type": {
#         "name": "ActionType",
#         "type": "struct",
#         "definition": {"fields": {"id": "Int64", "Type": "String", ...}},
#         "data": [...]
#       },
#       "uuid-for-outcome-categories": {
#         "name": "OutcomeCategories",
#         "type": "categorical",
#         "definition": {"categories": ["Action1", "Action2", "Action3"]}
#       }
#     },
#     "default": {...}
#   },
#   "roots": [...],
#   "parameters_col": "parameters",
#   "parameters": {},
#   "mode": "first_match",
#   "output_fn": null,
#   "post_process_fn": null,
#   "format_prioritized_fn": null
# }
