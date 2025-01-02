import { useRete } from "rete-react-plugin";
import { createEditor, Schemes, AreaExtra, Node, Connection } from "./config";
import { AreaPlugin } from "rete-area-plugin";
import { useEffect, useRef, useState } from "react";
import { useEditorDispatch, useEditor } from "@ctx/editor/EditorContext";
import { EditorState } from "@ctx/editor/editorTypes";
import { fetchProjectNodes } from "@ctx/editor/editorActions";
import { NodeEditor, BaseSchemes, getUID, ClassicPreset } from 'rete';
import {
  AutoArrangePlugin,
  Presets as ArrangePresets,
  ArrangeAppliers
} from "rete-auto-arrange-plugin";
import { ReadonlyPlugin } from "rete-readonly-plugin";




//@ts-ignore
const updateNodes = async (
  area: AreaPlugin<Schemes, AreaExtra>, 
  editor: NodeEditor<Schemes>, 
  socket: ClassicPreset.Socket,
  editorState: EditorState,
  rearrangeLayout: () => Promise<void>,
  readonly: ReadonlyPlugin<Schemes>,
) => {
  const createdNodes: Record<string, any> = {};
  // TODO dunamic update logic

  readonly.disable();
  await editor.clear();

  // Create Nodes
  const nodesToRemove: string[] = [];
  await Promise.all(Object.keys(editorState.nodes).map(async (nodeId) => {
    const nodeData = editorState.nodes[nodeId];
    let node: Node;
    if (nodeData.node_type === 'leaf') {
      node = new Node(`${nodeData.leaf_value}`);
      node.addInput("inputs", new ClassicPreset.Input(socket, "Inputs"))
      if (nodeData.left_child) {nodesToRemove.push(nodeData.left_child)}
      if (nodeData.right_child) {nodesToRemove.push(nodeData.right_child)}
    } else {
      node = new Node(`${nodeData.split_feature_id}`);
      node.addInput("inputs", new ClassicPreset.Input(socket, "Inputs"))
      node.addOutput(`output-right`, new ClassicPreset.Output(socket, 'Right Output'));
      node.addOutput(`output-left`, new ClassicPreset.Output(socket, 'Left Output'));
    }
    node.id = nodeId; 
    createdNodes[nodeId] = node;
    await editor.addNode(node);
  }));

  // Create Connections
  await Promise.all(nodesToRemove.map(async (nodeId) => {
    await editor.removeNode(nodeId);
  }))
  await Promise.all(Object.keys(editorState.nodes).map(async (nodeId) => {
    const nodeData = editorState.nodes[nodeId];
    if (nodeData.node_type === 'leaf') {return;}
    const parentNode = createdNodes[nodeId];
    if (nodeData.left_child){
      const childNode = createdNodes[nodeData.left_child];
      await editor.addConnection(
        new Connection(
          parentNode, `output-left`,
          childNode, "inputs", 
        ))
    }
    if (nodeData.right_child){
      const childNode = createdNodes[nodeData.right_child];
      await editor.addConnection(
        new Connection(
          parentNode, `output-right`,
          childNode, "inputs", 
        ))
    }
    }));

  await rearrangeLayout();
  readonly.enable();

  // Automatically rearrange objects
  
}

export default function ReteEditor(props: {}) {
  const [ref, reteEditor] = useRete(createEditor);
  const dispatch = useEditorDispatch();
  const editorState = useEditor();
  const currentProjectId = useRef("");

  useEffect(() => {
    if (currentProjectId.current !== editorState.projectId && editorState.projectId) {
      currentProjectId.current = editorState.projectId;
      fetchProjectNodes(editorState.projectId, dispatch)
    }
  }, [editorState.projectId, dispatch, currentProjectId]);

  useEffect(() => {
    reteEditor?.setEditorDispatch(dispatch)
  }, [reteEditor, dispatch])

  useEffect(() => {
    if (reteEditor !== null && editorState?.nodes){
      updateNodes(
        reteEditor.area,
        reteEditor.editor,
        reteEditor.socket,
        editorState,
        reteEditor.rearrangeLayout,
        reteEditor.readonly,
      )
    }
  }, [editorState?.nodes, reteEditor])
  

  return (
    <div ref={ref} className="flex flex-1 flex-col"></div>
  );
}
