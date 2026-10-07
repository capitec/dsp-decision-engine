import typing as t
import uuid
import os
import json
from datetime import datetime, timezone, timedelta

from sqlalchemy import select

from dsp_decider.cache import get_cache
from dsp_decider.cache.keys import CacheKeys
from dsp_decider.db.session import db_session
from dsp_decider.db.models.projects import ProjectRevision, RevisionType


class SessionManager:
    """Manages cache-based editing sessions for projects"""

    def __init__(self):
        self.cache = get_cache()
        self.instance_id = os.getenv("INSTANCE_ID", str(uuid.uuid4()))

    async def join_session(self, project_id: str, user_id: str) -> dict:
        """Join or create a session for a project"""
        session_key = CacheKeys.edit_session(project_id)

        # Check for existing session
        try:
            existing_session = await self.cache.get(session_key)
            if existing_session:
                # Add user to active users if not already present
                if user_id not in existing_session.get("active_users", []):
                    existing_session["active_users"].append(user_id)
                existing_session["last_activity_at"] = datetime.now(
                    timezone.utc
                ).isoformat()

                # Refresh TTL
                await self.cache.put(session_key, existing_session, 3600)
                return existing_session

        except KeyError:
            # Session doesn't exist, will create new one
            pass

        # Try to claim ownership of this project session
        await self._claim_session_ownership(project_id)

        # Get latest snapshot for this project
        latest_revision = await self._get_latest_revision(project_id)

        # Create new session
        session_data = {
            "session_id": str(uuid.uuid4()),
            "project_id": project_id,
            "active_users": [user_id],
            "instance_id": self.instance_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_activity_at": datetime.now(timezone.utc).isoformat(),
            "operation_count": 0,
            "current_snapshot_id": str(latest_revision.id) if latest_revision else None,
            "operations": [],
        }

        # Store session with 1 hour TTL
        await self.cache.put(session_key, session_data, 3600)

        # Register with active sessions and instance tracking
        await self._add_to_active_sessions(session_key)
        await self._add_to_instance_sessions(project_id)

        return session_data

    async def add_operation(
        self, project_id: str, user_id: str, operation: dict
    ) -> None:
        """Add an operation to the project session"""
        session_key = CacheKeys.edit_session(project_id)

        try:
            session_data = await self.cache.get(session_key)
            if not session_data:
                raise ValueError("No active session found for project")

            # Add user_id to operation if not present
            operation["user_id"] = user_id
            operation["timestamp"] = operation.get(
                "timestamp", int(datetime.now(timezone.utc).timestamp() * 1000)
            )

            # Add operation (keep last 100)
            session_data["operations"].append(operation)
            if len(session_data["operations"]) > 100:
                session_data["operations"].pop(0)  # Remove oldest

            session_data["operation_count"] += 1
            session_data["last_activity_at"] = datetime.now(timezone.utc).isoformat()

            # Ensure user is in active_users
            if user_id not in session_data.get("active_users", []):
                session_data["active_users"].append(user_id)

            # Update cache with fresh TTL
            await self.cache.put(session_key, session_data, 3600)

        except Exception as e:
            raise ValueError(f"Failed to add operation: {e}")

    async def recover_project_state(self, project_id: str) -> dict:
        """Recover project state from cache session or latest snapshot"""
        session_key = CacheKeys.edit_session(project_id)

        try:
            # Try to get active session from cache
            session_data = await self.cache.get(session_key)

            if session_data and session_data.get("operations"):
                # Active session with operations - reconstruct state from checkpoint
                operations = session_data.get("operations", [])

                # Find the most recent checkpoint operation
                checkpoint_revision = None
                for operation in reversed(operations):
                    if operation.get("operation_type") == "checkpoint_created":
                        checkpoint_id = operation.get("checkpoint_id")
                        if checkpoint_id:
                            checkpoint_revision = await self._get_revision_by_id(
                                checkpoint_id
                            )
                            break

                # If no checkpoint in operations, use session's current_snapshot_id
                if not checkpoint_revision and session_data.get("current_snapshot_id"):
                    checkpoint_revision = await self._get_revision_by_id(
                        session_data["current_snapshot_id"]
                    )

                # Use checkpoint as base state
                base_state = (
                    checkpoint_revision.full_state.copy() if checkpoint_revision else {}
                )

                # Apply operations that happened after the checkpoint
                # For now, we only have checkpoint operations, but in future we'll add:
                # move_node, change_threshold, etc.
                current_state = base_state

                return {
                    "state": current_state,
                    "session_data": session_data,
                    "recovered_from": "active_session_with_checkpoint",
                    "pending_operations": len(
                        [
                            op
                            for op in operations
                            if op.get("operation_type") != "checkpoint_created"
                        ]
                    ),
                    "checkpoint_id": (
                        str(checkpoint_revision.id) if checkpoint_revision else None
                    ),
                }

            elif session_data:
                # Active session but no operations - use current snapshot
                base_revision = await self._get_revision_by_id(
                    session_data["current_snapshot_id"]
                )
                return {
                    "state": base_revision.full_state if base_revision else {},
                    "session_data": session_data,
                    "recovered_from": "session_snapshot",
                }
            else:
                # No active session - load from latest snapshot
                latest_revision = await self._get_latest_revision(project_id)
                return {
                    "state": latest_revision.full_state if latest_revision else {},
                    "session_data": None,
                    "recovered_from": "latest_snapshot",
                }

        except Exception as e:
            # Fallback to latest snapshot
            latest_revision = await self._get_latest_revision(project_id)
            return {
                "state": latest_revision.full_state if latest_revision else {},
                "session_data": None,
                "recovered_from": "fallback_snapshot",
                "error": str(e),
            }

    async def update_heartbeat(self) -> None:
        """Update instance heartbeat to show we're alive"""
        heartbeat_key = CacheKeys.instance_heartbeat(self.instance_id)
        await self.cache.put(
            heartbeat_key, int(datetime.now(timezone.utc).timestamp() * 1000), 7200
        )

    # Private helper methods

    async def _claim_session_ownership(self, project_id: str) -> None:
        """Claim ownership of a project session, handling failover if needed"""
        owner_key = CacheKeys.session_owner(project_id)

        try:
            current_owner = await self.cache.get(owner_key)

            if current_owner and current_owner != self.instance_id:
                # Check if current owner is still alive
                heartbeat_key = CacheKeys.instance_heartbeat(current_owner)
                try:
                    last_heartbeat = await self.cache.get(heartbeat_key)
                    heartbeat_age = (
                        datetime.now(timezone.utc).timestamp() * 1000 - last_heartbeat
                    )

                    if heartbeat_age < 3600000:  # 1 hour in milliseconds
                        raise ValueError(
                            f"Project session owned by active instance: {current_owner}"
                        )

                    # Owner is dead, reassign
                    await self._reassign_dead_instance(current_owner)

                except KeyError:
                    # No heartbeat found, owner is dead
                    await self._reassign_dead_instance(current_owner)

        except KeyError:
            # No current owner
            pass

        # Claim ownership
        await self.cache.put(owner_key, self.instance_id, 3600)

    async def _reassign_dead_instance(self, dead_instance_id: str) -> None:
        """Reassign sessions from a dead instance to this instance"""
        try:
            dead_instance_key = CacheKeys.instance_sessions(dead_instance_id)
            dead_sessions = await self.cache.get(dead_instance_key, [])

            # Reassign each session to this instance
            for project_id in dead_sessions:
                owner_key = CacheKeys.session_owner(project_id)
                await self.cache.put(owner_key, self.instance_id, 3600)
                await self._add_to_instance_sessions(project_id)

            # Clean up dead instance data
            await self.cache.delete(dead_instance_key)
            await self.cache.delete(CacheKeys.instance_heartbeat(dead_instance_id))

        except Exception as e:
            # Log error but don't fail the operation
            print(f"Failed to reassign dead instance {dead_instance_id}: {e}")

    async def _add_to_active_sessions(self, session_key: str) -> None:
        """Add session to global active sessions list"""
        active_key = CacheKeys.active_sessions()
        try:
            active_sessions = await self.cache.get(active_key, [])
            if session_key not in active_sessions:
                active_sessions.append(session_key)
                await self.cache.put(active_key, active_sessions, 7200)
        except KeyError:
            await self.cache.put(active_key, [session_key], 7200)

    async def _add_to_instance_sessions(self, project_id: str) -> None:
        """Add project to this instance's session list"""
        instance_key = CacheKeys.instance_sessions(self.instance_id)
        try:
            instance_sessions = await self.cache.get(instance_key, [])
            if project_id not in instance_sessions:
                instance_sessions.append(project_id)
                await self.cache.put(instance_key, instance_sessions, 7200)
        except KeyError:
            await self.cache.put(instance_key, [project_id], 7200)

    async def _get_latest_revision(
        self, project_id: str
    ) -> t.Optional[ProjectRevision]:
        """Get the latest revision for a project"""
        try:
            async with db_session() as session:
                query = (
                    select(ProjectRevision)
                    .where(ProjectRevision.project_id == project_id)
                    .order_by(ProjectRevision.revision_number.desc())
                    .limit(1)
                )
                result = await session.execute(query)
                return result.scalar_one_or_none()
        except Exception as e:
            print(f"Error getting latest revision for project {project_id}: {e}")
            return None

    async def _get_revision_by_id(
        self, revision_id: str
    ) -> t.Optional[ProjectRevision]:
        """Get a specific revision by ID"""
        try:
            async with db_session() as session:
                query = select(ProjectRevision).where(ProjectRevision.id == revision_id)
                result = await session.execute(query)
                return result.scalar_one_or_none()
        except Exception as e:
            print(f"Error getting revision {revision_id}: {e}")
            return None

    async def _clear_session_operations(self, project_id: str) -> None:
        """Clear operations from session cache (used after checkpoint creation)"""
        try:
            session_key = CacheKeys.edit_session(project_id)
            session_data = await self.cache.get(session_key)

            if session_data:
                session_data["operations"] = []
                session_data["operation_count"] = 0
                session_data["last_activity_at"] = datetime.now(
                    timezone.utc
                ).isoformat()

                # Update cache with fresh TTL
                await self.cache.put(session_key, session_data, 3600)

        except Exception as e:
            print(f"Error clearing session operations for project {project_id}: {e}")

    async def _update_session_checkpoint(
        self, project_id: str, checkpoint_id: str
    ) -> None:
        """Update session to point to new checkpoint as base"""
        try:
            session_key = CacheKeys.edit_session(project_id)
            session_data = await self.cache.get(session_key)

            if session_data:
                session_data["current_snapshot_id"] = checkpoint_id
                session_data["last_activity_at"] = datetime.now(
                    timezone.utc
                ).isoformat()

                # Update cache with fresh TTL
                await self.cache.put(session_key, session_data, 3600)

        except Exception as e:
            print(f"Error updating session checkpoint for project {project_id}: {e}")
