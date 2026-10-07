import typing as t
from datetime import datetime, timedelta, timezone
from fastapi import Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, func

from dsp_decider.db.session import db_session
from dsp_decider.db.models.projects import Project, ProjectShare, AccessMode
from dsp_decider.db.models.users import Team
from dsp_decider.data_sources import DataSourceManager
from dsp_decider.data_sources.core import DataSourceInfo
from dsp_decider.settings import settings
from dsp_decider.auth.authorization import get_current_user_session, is_superadmin
from dsp_decider.auth.models import SessionInfo
from dsp_decider.api.v1.project.project import ProjectResponse
from .routes import router


class ActivitySummary(BaseModel):
    """Activity summary for different time periods"""

    today: int
    this_week: int
    this_month: int


class OverviewResponse(BaseModel):
    """Overview dashboard response with statistics and recent activity"""

    model_config = ConfigDict(from_attributes=True)

    # Stats
    total_projects: int
    total_datasources: int
    recent_activity_count: int  # Projects updated in last 7 days

    # Recent projects (last 5, sorted by updated_at)
    recent_projects: t.List[ProjectResponse]

    # Available datasources
    datasources: t.List[DataSourceInfo]

    # Activity timeline
    activity_summary: ActivitySummary


def get_datasource_manager() -> DataSourceManager:
    """Get data source manager instance"""
    return DataSourceManager(settings.datasources)


@router.get("", response_model=OverviewResponse)
async def get_overview(
    current_user: SessionInfo = Depends(get_current_user_session),
    manager: DataSourceManager = Depends(get_datasource_manager),
) -> OverviewResponse:
    """Get dashboard overview with statistics and recent activity"""

    async with db_session() as session:
        # Calculate time boundaries
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        month_start = now - timedelta(days=30)

        # Get total project count
        total_count_query = select(func.count(Project.id))
        total_result = await session.execute(total_count_query)
        total_projects = total_result.scalar() or 0

        # Get activity counts for different time periods
        today_count_query = select(func.count(Project.id)).where(
            Project.updated_at >= today_start
        )
        today_result = await session.execute(today_count_query)
        today_count = today_result.scalar() or 0

        week_count_query = select(func.count(Project.id)).where(
            Project.updated_at >= week_start
        )
        week_result = await session.execute(week_count_query)
        week_count = week_result.scalar() or 0

        month_count_query = select(func.count(Project.id)).where(
            Project.updated_at >= month_start
        )
        month_result = await session.execute(month_count_query)
        month_count = month_result.scalar() or 0

        # Get recent projects (last 5, sorted by updated_at)
        # Need to construct ProjectResponse with access_mode and is_owner
        recent_project_responses = []

        if await is_superadmin(current_user):
            # Superadmins see all projects
            recent_query = select(Project).order_by(Project.updated_at.desc()).limit(5)
            recent_result = await session.execute(recent_query)
            recent_projects = recent_result.scalars().all()

            for project in recent_projects:
                is_owner = project.created_by == current_user.user_id
                recent_project_responses.append(
                    ProjectResponse(
                        **project.__dict__,
                        is_owner=is_owner,
                        access_mode=AccessMode.ADMIN,
                    )
                )
        else:
            # Get user's owned projects
            owned_query = (
                select(Project)
                .where(Project.created_by == current_user.user_id)
                .order_by(Project.updated_at.desc())
                .limit(5)
            )
            owned_result = await session.execute(owned_query)
            owned_projects = owned_result.scalars().all()

            for project in owned_projects:
                recent_project_responses.append(
                    ProjectResponse(
                        **project.__dict__,
                        is_owner=True,
                        access_mode=AccessMode.ADMIN,
                    )
                )

            # If we don't have 5 yet, get some shared projects
            if len(recent_project_responses) < 5:
                remaining = 5 - len(recent_project_responses)
                shared_query = (
                    select(Project, ProjectShare.access_mode)
                    .join(ProjectShare)
                    .where(ProjectShare.user_id == current_user.user_id)
                    .order_by(Project.updated_at.desc())
                    .limit(remaining)
                )
                shared_result = await session.execute(shared_query)
                shared_projects = shared_result.all()

                for project, access_mode in shared_projects:
                    recent_project_responses.append(
                        ProjectResponse(
                            **project.__dict__,
                            is_owner=False,
                            access_mode=access_mode,
                        )
                    )

        # Sort by updated_at to get the most recent across owned and shared
        recent_project_responses.sort(key=lambda p: p.updated_at, reverse=True)

    # Get datasource information
    datasource_list = manager.list_datasources()

    return OverviewResponse(
        total_projects=total_projects,
        total_datasources=len(datasource_list.datasources),
        recent_activity_count=week_count,  # Projects updated in last 7 days
        recent_projects=recent_project_responses,
        datasources=datasource_list.datasources,
        activity_summary=ActivitySummary(
            today=today_count,
            this_week=week_count,
            this_month=month_count,
        ),
    )
