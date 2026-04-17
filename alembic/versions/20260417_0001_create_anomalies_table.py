"""create anomalies table

Revision ID: 20260417_0001
Revises:
Create Date: 2026-04-17 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260417_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "anomalies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("cluster_key", sa.String(length=128), nullable=False),
        sa.Column("ai_summary", sa.Text(), nullable=False),
        sa.Column("root_cause_hypothesis", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("jira_issue_key", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_anomalies_cluster_key"), "anomalies", ["cluster_key"], unique=True)
    op.create_index(op.f("ix_anomalies_jira_issue_key"), "anomalies", ["jira_issue_key"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_anomalies_jira_issue_key"), table_name="anomalies")
    op.drop_index(op.f("ix_anomalies_cluster_key"), table_name="anomalies")
    op.drop_table("anomalies")
