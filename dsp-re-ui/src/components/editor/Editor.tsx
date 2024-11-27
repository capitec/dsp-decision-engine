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

  // Create Nodes
  await Promise.all(editorState.nodes.map(async (nodeData) => {
    const { name, dependencies } = nodeData;
    const node = new Node(name);
    node.id = getUID(); 

    node.addOutput('output', new ClassicPreset.Output(socket, 'Output'));

    dependencies.forEach(element => {
      node.addInput(element, new ClassicPreset.Input(socket, element))
    });

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
            node, depName
          ))
      }
    }));
  }))

  await rearrangeLayout();

  // Automatically rearange objects
  
}

export default function ReteEditor(props: {projectId: string}) {
  const [ref, reteEditor] = useRete(createEditor);
  const dispatch = useEditorDispatch();
  const editorState = useEditor();
  const currentProjectId = useRef("");

  useEffect(() => {
    if (currentProjectId.current !== props.projectId) {
      currentProjectId.current = props.projectId;
      fetchProjectNodes(props.projectId, dispatch)
    }
  }, [props.projectId, dispatch, currentProjectId]);


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
  }, [editorState, reteEditor])
  

  return (
    <div ref={ref} style={{ height: "100vh", width: "100vw" }}></div>
  );
}
