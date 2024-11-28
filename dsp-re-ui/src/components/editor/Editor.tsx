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




//@ts-ignore
const updateNodes = async (
  area: AreaPlugin<Schemes, AreaExtra>, 
  editor: NodeEditor<Schemes>, 
  socket: ClassicPreset.Socket,
  editorState: EditorState,
  rearrangeLayout: () => Promise<void>,
) => {
  const createdNodes: Record<string, any> = {};
  // TODO dunamic update logic
  await editor.clear();

  // Create Nodes
  await Promise.all(editorState.nodes.map(async (nodeData) => {
    const { name, dependencies, nodeId } = nodeData;
    const node = new Node(name);
    node.id = nodeId; 

    node.addOutput('output', new ClassicPreset.Output(socket, 'Output'));
    if (dependencies.length > 0) {
      node.addInput("inputs", new ClassicPreset.Input(socket, "Inputs"))
    }

    createdNodes[name] = node;
    await editor.addNode(node);
  }));

  // Create Connections
  await Promise.all(editorState.nodes.map(async (nodeData) => {
    const { name, dependencies } = nodeData;
    const node = createdNodes[name];

    await Promise.all(dependencies.map(async (depName) => {
      const dependencyNode = createdNodes[depName];
      if (dependencyNode) {
        await editor.addConnection(
          new Connection(
            dependencyNode, "output", 
            node, "inputs"
          ))
      }
    }));
  }))

  await rearrangeLayout();

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
      )
    }
  }, [editorState?.nodes, reteEditor])
  

  return (
    <div ref={ref} style={{ height: "100vh", width: "100vw" }}></div>
  );
}
