"""Add model_id and prediction_label to user_requests

Revision ID: a1b2c3d4e5f6
Revises: fc6240d91fcd
Create Date: 2026-05-28 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'fc6240d91fcd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add model_id and prediction_label columns."""
    op.add_column(
        'user_requests',
        sa.Column('prediction_label', sa.String(length=100), nullable=True)
    )
    op.add_column(
        'user_requests',
        sa.Column('model_id', sa.String(length=255), nullable=True)
    )
    op.create_index(
        op.f('ix_user_requests_prediction_label'),
        'user_requests', ['prediction_label'], unique=False
    )
    op.create_index(
        op.f('ix_user_requests_model_id'),
        'user_requests', ['model_id'], unique=False
    )


def downgrade() -> None:
    """Remove model_id and prediction_label columns."""
    op.drop_index(op.f('ix_user_requests_model_id'), table_name='user_requests')
    op.drop_index(op.f('ix_user_requests_prediction_label'), table_name='user_requests')
    op.drop_column('user_requests', 'model_id')
    op.drop_column('user_requests', 'prediction_label')
