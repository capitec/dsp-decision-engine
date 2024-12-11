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
  await Promise.all(Object.keys(editorState.nodes).map(async (nodeId) => {
    const nodeData = editorState.nodes[nodeId];
    if ("condition_type" in nodeData){
      const { condition, connections } = nodeData;
      const node = new Node(condition);
      node.id = nodeId; 
  
      connections.map((v,i)=>{
        node.addOutput(`output-${i}`, new ClassicPreset.Output(socket, 'Output'));
      })
      if (nodeId !== "*root*"){
        node.addInput("inputs", new ClassicPreset.Input(socket, "Inputs"))
      }
  
      createdNodes[nodeId] = node;
      await editor.addNode(node);
    } else {
      const node = new Node("value");
      node.id = nodeId; 
      node.addInput("inputs", new ClassicPreset.Input(socket, "Inputs"))
      createdNodes[nodeId] = node;
      await editor.addNode(node);

    }
  }));

  // Create Connections
  await Promise.all(Object.keys(editorState.nodes).map(async (nodeId) => {
    const nodeData = editorState.nodes[nodeId];

    if ("condition_type" in nodeData){
      const { connections } = nodeData;
      const parentNode = createdNodes[nodeId];

      await Promise.all(connections.map(async (conn,connIdx) => {
        await Promise.all(conn.values.map(async (value) => {
          const childNode = createdNodes[value];
          if (childNode) {
            await editor.addConnection(
              new Connection(
                parentNode, `output-${connIdx}`,
                childNode, "inputs", 
              ))
          }
        }))
      }));
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
