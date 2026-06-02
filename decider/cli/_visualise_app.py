"""
Streamlit app — launched by `decider visualise`.

Env vars:
    DECIDER_VISUALISE_PROJECT_DIR
    DECIDER_VISUALISE_EXT_DIR       (optional)
    DECIDER_VISUALISE_CONFIG_DIR    (optional)
    DECIDER_VISUALISE_ROOT_MODULE   (optional, default "main")
"""

import json
import os
import sys
from pathlib import Path

import polars as pl
import streamlit as st

# ── bootstrap ─────────────────────────────────────────────────────────────────

_project_dir = Path(os.environ.get("DECIDER_VISUALISE_PROJECT_DIR", ".")).resolve()
_repo_root = _project_dir.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_ext_dir = os.environ.get("DECIDER_VISUALISE_EXT_DIR",
                           str(_project_dir / "decider_extensions"))
_configs_dir = os.environ.get("DECIDER_VISUALISE_CONFIG_DIR",
                               str(_project_dir / "configs"))
_root_module = os.environ.get("DECIDER_VISUALISE_ROOT_MODULE", "main")

from decider.initialization import initialize_decider
from decider.config.file import JsonFileConfigManager
from decider.modules import GraphModule
from decider.cli._graph import (
    build_graph,
    build_expression_graph,
    run_with_intermediates,
    _kind,
)

# ── page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Decider Visualise",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state init ────────────────────────────────────────────────────────

if "breadcrumb" not in st.session_state:
    # Each entry: {"label": str, "module": BaseModule}
    st.session_state.breadcrumb = []

if "run_inputs" not in st.session_state:
    st.session_state.run_inputs = None   # Dict[str, pl.DataFrame] when set

# ── load root module (cached) ─────────────────────────────────────────────────

@st.cache_resource
def _load_root(root_key: str):
    initialize_decider(extension_path=_ext_dir)
    import asyncio
    mgr = JsonFileConfigManager(basepath=_configs_dir)
    versioned = asyncio.run(mgr.get_latest())
    module = GraphModule.model_validate(versioned.config[root_key]).root
    return module, versioned


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Project")
    st.caption(str(_project_dir))

    root_key = st.text_input("Root module key", value=_root_module)
    if st.button("↺  Reload config"):
        st.cache_resource.clear()
        st.session_state.breadcrumb = []
        st.session_state.run_inputs = None
        st.rerun()

try:
    root_module, versioned = _load_root(root_key)
except Exception as e:
    st.error(f"Could not load module: {e}")
    st.stop()

with st.sidebar:
    st.divider()
    st.caption(f"version  {versioned.version}")
    st.caption(f"type     {root_module.type}")

    # ── input data entry ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Run data")
    st.caption("Paste JSON (column-oriented) to push data through the pipeline.")

    default_cols = root_module.get_input_frame_keys()
    json_placeholder = json.dumps(
        {k: ["value1", "value2"] for k in
         (root_module._compute_input_frame_keys() if hasattr(root_module, '_compute_input_frame_keys') else ["input"])},
        indent=2,
    )
    raw_json = st.text_area("Input JSON", value="", height=180,
                             placeholder=json_placeholder)
    if st.button("▶  Run"):
        try:
            parsed = json.loads(raw_json)
            # support both {col: [...]} (single frame) and {"frame": {col: [...]}}
            if parsed and isinstance(next(iter(parsed.values())), dict):
                st.session_state.run_inputs = {
                    k: pl.DataFrame(v) for k, v in parsed.items()
                }
            else:
                st.session_state.run_inputs = {"input": pl.DataFrame(parsed)}
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    if st.session_state.run_inputs is not None:
        if st.button("✕  Clear run"):
            st.session_state.run_inputs = None
            st.rerun()


# ── breadcrumb navigation ─────────────────────────────────────────────────────

# current module is root unless the user has drilled in
crumb_stack = st.session_state.breadcrumb
current_module = crumb_stack[-1]["module"] if crumb_stack else root_module

# render breadcrumb bar
crumb_parts = [{"label": root_key, "module": root_module}] + crumb_stack
cols = st.columns([1] * len(crumb_parts) + [8])
for i, crumb in enumerate(crumb_parts):
    with cols[i]:
        is_last = i == len(crumb_parts) - 1
        if is_last:
            st.markdown(f"**{crumb['label']}**")
        else:
            if st.button(crumb["label"], key=f"crumb_{i}"):
                st.session_state.breadcrumb = crumb_stack[: i]   # pop back to i
                st.rerun()

if crumb_stack:
    st.caption(f"type: {current_module.type}  ·  name: {current_module.name}")

st.divider()

# ── main content: tabs ────────────────────────────────────────────────────────

tab_graph, tab_run, tab_config = st.tabs(["Graph", "Run output", "Config"])

# ── TAB: Graph ────────────────────────────────────────────────────────────────

with tab_graph:
    kind = _kind(current_module)

    if kind == "expression":
        # show the intra-module expression DAG
        st.caption("Expression node DAG — functions, column inputs and config injections")
        eg = build_expression_graph(current_module)
        dot = eg.to_graphviz()
        st.graphviz_chart(dot.source, use_container_width=True)

        # node table
        rows = [{"node": n.label, "kind": n.kind,
                 **{f"cfg:{k}": v for k, v in n.fields.items()}}
                for n in eg.nodes]
        if rows:
            st.dataframe(pl.DataFrame(rows, infer_schema_length=len(rows)),
                         use_container_width=True)

    else:
        # show the module-level structural graph
        g = build_graph(current_module)
        col_g, col_d = st.columns([2, 1])

        with col_g:
            dot = g.to_graphviz()
            st.graphviz_chart(dot.source, use_container_width=True)

        with col_d:
            st.subheader("Modules")
            for n in g.nodes:
                if not n.drillable:
                    continue
                c1, c2 = st.columns([4, 1])
                with c1:
                    tag = f"`{n.type_id}`"
                    cfg = "  ·  " + "  ".join(f"{k}={v}" for k, v in n.fields.items()) if n.fields else ""
                    st.markdown(f"**{n.label}** {tag}{cfg}")
                with c2:
                    if st.button("→", key=f"drill_{n.id}",
                                 help=f"Drill into {n.label}"):
                        st.session_state.breadcrumb = crumb_stack + [
                            {"label": n.label, "module": n.module_ref}
                        ]
                        st.rerun()

# ── TAB: Run output ───────────────────────────────────────────────────────────

with tab_run:
    if st.session_state.run_inputs is None:
        st.info("Paste input data in the sidebar and click **▶ Run** to see intermediate outputs.")
    else:
        inputs = st.session_state.run_inputs

        st.subheader("Input")
        for frame_key, df in inputs.items():
            st.caption(f"frame: `{frame_key}`")
            st.dataframe(df, use_container_width=True)

        st.subheader("Intermediates")
        try:
            intermediates = run_with_intermediates(current_module, inputs)
        except Exception as e:
            st.error(f"Execution error: {e}")
            intermediates = []

        for label, df in intermediates:
            with st.expander(f"after  **{label}**", expanded=True):
                # highlight newly-added columns vs the input
                input_cols = set(next(iter(inputs.values())).columns)
                new_cols = [c for c in df.columns if c not in input_cols]
                st.caption(f"new columns: {', '.join(new_cols) if new_cols else '(none)'}")
                st.dataframe(df, use_container_width=True)

# ── TAB: Config ───────────────────────────────────────────────────────────────

with tab_config:
    try:
        st.json(current_module.model_dump())
    except Exception:
        st.json(versioned.config)
