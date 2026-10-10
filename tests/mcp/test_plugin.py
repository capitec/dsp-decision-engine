from __future__ import annotations

import json
import re
from pathlib import Path

from decikit._tooling.mcp import Policy, build_server
from decikit._tooling.mcp._editor import EditorBridge
from tests.mcp.test_mcp import _tools

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins" / "decikit"
SKILL = PLUGIN / "skills" / "decikit"
REFERENCES = SKILL / "references"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _frontmatter(path: Path) -> dict[str, str]:
    head = path.read_text().split("---")[1]
    return dict(line.split(": ", 1) for line in head.strip().splitlines())


def _tool_names(path: Path) -> list[str]:
    section = re.search(r"^## Tools\n(.*?)(?=^## |\Z)", path.read_text(), re.M | re.S)
    return re.findall(r"^- `(\w+)[(`]", section.group(1), re.M) if section else []


def test_marketplace_names_the_plugin_directory():
    market = _json(ROOT / ".claude-plugin" / "marketplace.json")
    assert market["name"] and market["owner"]["name"]
    (entry,) = market["plugins"]
    source = ROOT / entry["source"]
    assert ".." not in entry["source"] and source.is_dir()
    assert entry["name"] == _json(source / ".claude-plugin" / "plugin.json")["name"]


def test_plugin_manifest_has_the_required_fields_and_no_pinned_version():
    plugin = _json(PLUGIN / ".claude-plugin" / "plugin.json")
    assert plugin["name"] == "decikit"
    assert {"description", "author", "repository", "license"} <= set(plugin)
    assert "version" not in plugin


def test_mcp_server_runs_decikit_mcp_with_fastmcp():
    server = _json(PLUGIN / ".mcp.json")["mcpServers"]["decikit"]
    assert [server["command"], *server["args"]][-2:] == ["decikit", "mcp"]
    assert "fastmcp>=4.0.10" in server["args"]


def test_skill_is_named_and_described_for_triggering():
    meta = _frontmatter(SKILL / "SKILL.md")
    assert meta["name"] == "decikit"
    assert all(word in meta["description"] for word in ("flow", "scorecard", "VS Code"))


def test_skill_links_every_reference_and_every_link_exists():
    linked = set(re.findall(r"\]\(references/([\w-]+\.md)", (SKILL / "SKILL.md").read_text()))
    present = {p.name for p in REFERENCES.glob("*.md")}
    assert not linked - present, f"linked from SKILL.md but missing: {sorted(linked - present)}"
    assert not present - linked, f"in references/ but not linked from SKILL.md: {sorted(present - linked)}"


def test_references_link_only_to_files_that_exist():
    broken = [
        f"{page.name} -> {target}"
        for page in REFERENCES.glob("*.md")
        for target in re.findall(r"\]\(([\w-]+\.md)", page.read_text())
        if not (REFERENCES / target).exists()
    ]
    assert not broken, f"broken reference links: {broken}"


def test_tools_sections_name_exactly_the_registered_tools(tmp_path):
    registered = set(_tools(build_server(Policy(), EditorBridge(tmp_path))))
    listed = {page: _tool_names(page) for page in [SKILL / "SKILL.md", *sorted(REFERENCES.glob("*.md"))]}
    named = {name for names in listed.values() for name in names}
    unregistered = {page.name: sorted(set(names) - registered) for page, names in listed.items() if set(names) - registered}
    assert not unregistered, f"listed in ## Tools but not registered: {unregistered}"
    assert not registered - named, f"registered but in no ## Tools section: {sorted(registered - named)}"
