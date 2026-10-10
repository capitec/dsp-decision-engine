"""Teams cannot be created by an ordinary user (superadmins only), so the sharing tests put them in the database
of the stack that `stack.sh` started (the container is named after E2E_PG_PORT)."""

import os
import subprocess
import uuid

import pytest


def _psql(sql: str) -> str:
    runtime = os.environ.get("E2E_RUNTIME", "podman")
    container = f"tree-editor-e2e-pg-{os.environ.get('E2E_PG_PORT', '55432')}"
    done = subprocess.run([runtime, "exec", container, "psql", "-U", "postgres", "-v", "ON_ERROR_STOP=1", "-tA", "-c", sql],
                          capture_output=True, text=True)
    if done.returncode != 0:
        pytest.skip(f"cannot seed teams: no database container {container} ({done.stderr.strip()[:100]})")
    return done.stdout.strip()


def create_team(name: str, owner_id: str, member_ids: tuple[str, ...] = ()) -> str:
    """Insert a team owned by `owner_id` (also a member with the owner role) and the plain members; returns its id."""
    team = str(uuid.uuid4())
    memberships = [(owner_id, "OWNER"), *((member, "MEMBER") for member in member_ids)]
    rows = ", ".join(f"('{uuid.uuid4()}', '{user}', '{team}', '{role}', now())" for user, role in memberships)
    _psql(f"INSERT INTO teams (id, name, created_by, created_at, updated_at) VALUES ('{team}', '{name}', '{owner_id}', now(), now()); "
          f"INSERT INTO team_memberships (id, user_id, team_id, role, joined_at) VALUES {rows}")
    return team
