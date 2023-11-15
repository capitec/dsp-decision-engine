#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Sholto.
# Distributed under the terms of the Modified BSD License.

"""
TODO: Add module docstring
"""

from ipywidgets import DOMWidget
from traitlets import Unicode, Int, List, Dict
from ._frontend import module_name, module_version


class LiteGraphWidget(DOMWidget):
    """TODO: Add docstring here
    """
    _model_name = Unicode('LiteGraphModel').tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    _view_name = Unicode('LiteGraphView').tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)

    width = Int(700).tag(sync=True)
    height = Int(400).tag(sync=True)
    graph = Dict(per_key_traits={
        "nodes": Dict(value_trait=Dict({
            "node_type": Unicode(),
            "title": Unicode(),
            "inputs": List(Unicode()),
            "outputs": List(Unicode()),
            "x": Int(),
            "y": Int(),
        })),
        "connections": List(value_trait=Dict({
            "from_node": Unicode(),
            "output_id": Int(),
            "to_node": Unicode(),
            "input_id": Int(),
        }))
    }).tag(sync=True)
