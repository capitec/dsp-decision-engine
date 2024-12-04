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



// //@ts-ignore
// const updateDecisionTree = async (
//   area: AreaPlugin<Schemes, AreaExtra>, 
//   editor: NodeEditor<Schemes>, 
//   socket: ClassicPreset.Socket,
//   decisionTreeData: any, // Replace 'any' with more specific type if possible
//   rearrangeLayout: () => Promise<void>,
// ) => {
//   const createdNodes: Record<string, any> = {};

//   // Clear existing editor
//   await editor.clear();

//   const renderTable = async (nodeData: any, parentNode: any, parentCondition: string) => {
//     console.log({type: "table",nodeData})
//     const nodeId =  nodeData.condition_table;
//     const node = new Node(nodeId);
//     node.id = nodeId;
//     nodeData.values && nodeData.values.map((k: any,i: any) => {
//       node.addOutput(`output-${i}`, new ClassicPreset.Output(socket, `Output ${i}`));
//     })
//     node.addInput('inputs', new ClassicPreset.Input(socket, 'Inputs'));
//     await editor.addNode(node);
//     await editor.addConnection(
//       new Connection(
//         parentNode, parentCondition, 
//         node, "inputs"
//       )
//     );
//     console.log({nodeData})
//     nodeData.values && await Promise.all(nodeData.values.map(async (v: any,i: any) => {
//       if (v.nodes){
//         await createNodes(v.nodes, node, `output-${i}`);
//       } else {
//         await renderValue(v, node, `output-${i}`);
//       }
//     }))

//   }

//   const renderValue = async (nodeData: any, parentNode: any, parentCondition: string) => {
//     console.log({nodeData, renderval: 1})
//     const nodeId = getUID();
//     const node = new Node(nodeId);
//     node.id = nodeId;
//     // node.addOutput('output', new ClassicPreset.Output(socket, 'Output'));
//     node.addInput('inputs', new ClassicPreset.Input(socket, 'Inputs'));

//     await editor.addNode(node);
//     await editor.addConnection(
//       new Connection(
//         parentNode, parentCondition, 
//         node, "inputs"
//       )
//     );
//   }

//   const renderBaseType = async (nodeData: any, parentNode: any, parentCondition: string) => {
//     const nodeId = nodeData.condition;
//     const node = new Node(nodeId);
//     node.id = nodeId;
//     node.addOutput('output', new ClassicPreset.Output(socket, 'Output'));
//     node.addInput('inputs', new ClassicPreset.Input(socket, 'Inputs'));

//     await editor.addNode(node);

//     await editor.addConnection(
//       new Connection(
//         parentNode, parentCondition, 
//         node, "inputs"
//       )
//     );

//     if (nodeData.value?.nodes && nodeData.value.nodes.length > 0) {
//       await createNodes(nodeData.value.nodes, node, "output");
//     } else if (nodeData.value) {
//       await renderValue(nodeData.value, node, "output");
//     }
//   }

//   // Recursive function to create nodes and connections
//   const createNodes = async (nodes: any[], parentNode: any, parentCondition: string) => {
//     for (const nodeData of nodes) {
//       if (nodeData.condition_type == "base"){
//         await renderBaseType(nodeData, parentNode, parentCondition)
//       } else if (nodeData.condition_type == "table"){
//         await renderTable(nodeData, parentNode, parentCondition)
//       } else if (nodeData.type == "DataFrame"){
//         await renderValue(nodeData, parentNode, parentCondition)
//       }
//     }
//   };

//   if (decisionTreeData.root?.nodes) {
//     const node = new Node("root");
//     node.id = "root";
//     node.addOutput('output', new ClassicPreset.Output(socket, 'Output'));
//     await editor.addNode(node);
//     await createNodes(decisionTreeData.root.nodes, node, "output");
//     await rearrangeLayout();
//   }
// };


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
      const { condition, values } = nodeData;
      const node = new Node(condition);
      node.id = nodeId; 
  
      node.addOutput('output', new ClassicPreset.Output(socket, 'Output'));
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
      const { values } = nodeData;
      const parentNode = createdNodes[nodeId];

      await Promise.all(values.map(async (value) => {
        const childNode = createdNodes[value];
        if (childNode) {
          await editor.addConnection(
            new Connection(
              parentNode, "output",
              childNode, "inputs", 
            ))
        }
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
    <div ref={ref} style={{ height: "100vh", width: "100vw" }}></div>
  );
}
