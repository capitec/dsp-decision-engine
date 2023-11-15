// Copyright (c) Sholto
// Distributed under the terms of the Modified BSD License.

import {
  DOMWidgetModel,
  DOMWidgetView,
  ISerializers,
} from '@jupyter-widgets/base';

import {
  LGraphCanvas,
  LGraph,
  LiteGraph,
  LGraphNode
} from "litegraph.js"

import { MODULE_NAME, MODULE_VERSION } from './version';

// Import the CSS
import '../css/widget.css';

export class LiteGraphModel extends DOMWidgetModel {
  defaults() {
    return {
      ...super.defaults(),
      _model_name: LiteGraphModel.model_name,
      _model_module: LiteGraphModel.model_module,
      _model_module_version: LiteGraphModel.model_module_version,
      _view_name: LiteGraphModel.view_name,
      _view_module: LiteGraphModel.view_module,
      _view_module_version: LiteGraphModel.view_module_version,
      width: 700,
      height: 400,
      graph: {}
    };
  }

  static serializers: ISerializers = {
    ...DOMWidgetModel.serializers,
    // Add any extra serializers here
  };

  static model_name = 'LiteGraphModel';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'LiteGraphView'; // Set to null if no view
  static view_module = MODULE_NAME; // Set to null if no view
  static view_module_version = MODULE_VERSION;
}

class StepNode extends LGraphNode {
  constructor(title?: string) {
    super(title);
    // this.addOutput("Result","Any");
  }
}

export class LiteGraphView extends DOMWidgetView {
  canvas: HTMLCanvasElement;
  lGraph: LGraph;
  liteGraphCanvas: LGraphCanvas

  initNodeTypes(){
    LiteGraph.clearRegisteredTypes();
    LiteGraph.registerNodeType("step", StepNode);

  }

  drawGraph() {
    const graph = this.model.get("graph");
    // console.log({graph});
    let lGraphNodes: any = {};
    const nodes = graph["nodes"];
    Object.keys(nodes).forEach(k => {
      console.log({k});
      const node = nodes[k];
      // console.log({node});
      const gNode = LiteGraph.createNode(node["node_type"]);
      // gNode.addInputs(node["inputs"].map((v: string) => [v, "Any"]));
      node["inputs"].forEach((v: string) => gNode.addInput(v, "Any"));
      node["outputs"].forEach((v: string) => gNode.addOutput(v, "Any"));
      gNode.pos = [node["x"], node["y"]];
      if (node["title"] !== undefined){
        gNode.title = node["title"]
      }
      this.lGraph.add(gNode)
      lGraphNodes[k] = gNode;
    })
    graph["connections"].forEach((el: any) => {
      console.log({el, lGraphNodes, fnode: el["from_node"], tnode:el["to_node"], from_node:lGraphNodes[el["from_node"]], toNode:lGraphNodes[el["to_node"]]});
      lGraphNodes[el["from_node"]].connect(el["output_id"], lGraphNodes[el["to_node"]], el["input_id"]);
    });
  }

  render() {

    // Setup Canvas
    this.canvas = document.createElement('canvas');
    this.setElement(this.canvas);
    this.lGraph = new LGraph();
    this.liteGraphCanvas = new LGraphCanvas(this.canvas, this.lGraph);
    this.resizeCanvas();
    this.model.on_some_change(['width', 'height'], this.resizeCanvas, this);
    console.log({lgraph: this.lGraph});
    this.initNodeTypes();
    this.drawGraph();
    // var node_const = LiteGraph.createNode("basic/const");
    // node_const.pos = [200,200];
    // graph.add(node_const);
    // node_const.setValue(4.5);

    // var node_watch = LiteGraph.createNode("basic/watch");
    // node_watch.pos = [700,200];
    // graph.add(node_watch);

    // node_const.connect(0, node_watch, 0 );

    this.lGraph.start()

    // this.el.classList.add('custom-widget');

    // this.value_changed();
    // this.model.on('change:value', this.value_changed, this);
  }


  resizeCanvas() {
    const width = this.model.get('width');
    const height = this.model.get('height');
    this.canvas.setAttribute('width', width);
    this.canvas.setAttribute('height', height);
    this.liteGraphCanvas.resize(width,height);
  }

  // value_changed() {
  //   this.el.textContent = this.model.get('value');
  // }
}
