import React, { useMemo, useState, useCallback, useEffect } from 'react';

import {
    Sidebar,
    SidebarContent,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarHeader,
    SidebarRail,
  } from "@/components/ui/sidebar"

import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import { 
  Label 
} from "@/components/ui/label";

import { ReactFlow, useOnSelectionChange } from '@xyflow/react';
import { Node, Edge, TreeNode, NumericalNodeData, CategoricalNodeData, LeafNodeData, RangeNodeData } from "../editor/types"
import { useFeatures, useNodes } from '@components/editor/EditorContext';
import { updateNodeData } from '@components/editor/util';
import { CategoricalConfig, LeafConfig, NumericalConfig, RangeConfig } from './Editors';
import { ProjectSidebar } from './ProjectEditor';

function useSelectedNode() {
    const [selectedNodeId, setSelectedNodeId] = useState<string|null>(null);
    const [nodes, setNodes, onNodesChange] = useNodes();

    const onChange = useCallback(({ nodes }: {nodes: Node[], edges: Edge[]}) => {
        if (nodes.length) {
            setSelectedNodeId(nodes[0].id)
        } else {
            setSelectedNodeId(null)
        }
    }, []);
    const selectedNode = useMemo(()=>{
        if (!selectedNodeId) { return null }
        return nodes.find(n=>n.id === selectedNodeId)
    }, [nodes, selectedNodeId])
     
    useOnSelectionChange({
        //@ts-ignore
        onChange, 
    });
    return selectedNode
}

export function AppSidebar() {
    const selectedNode = useSelectedNode();
    const [nodes, setNodes, onNodesChange] = useNodes();
    const [features, setFeatures] = useFeatures();

    const updateNode = useCallback(
        (value: Partial<TreeNode>, feats?: string[]) => setNodes(vals => {
            console.log({value, feats})
            return updateNodeData(selectedNode!, value, feats||features, vals)
        }),
        [selectedNode, features, setNodes],
    );

    const renderValueConfig = () => {
        return (<div>Value</div>)
    }

    const renderConfig = () => {
        if (!selectedNode) {
            return (<>
                <SidebarHeader>
                Project Configuration
                </SidebarHeader>
                <SidebarContent>
                <ProjectSidebar/>
                </SidebarContent>
            </>
        )
        }
        return (<>
            <SidebarHeader>
            Node Configuration
            </SidebarHeader>
            <SidebarContent>
            <SidebarGroup>
            <SidebarGroupContent className="space-y-4 p-4">
                {/* Condition Type Dropdown */}
                <div className="space-y-2">
                <Label>Condition Type</Label>
                <Select 
                    value={selectedNode.data.node_type} 
                    onValueChange={(node_type: TreeNode["node_type"]) => updateNode({node_type})}
                >
                    <SelectTrigger>
                    <SelectValue placeholder="Select condition type" />
                    </SelectTrigger>
                    <SelectContent>
                    <SelectItem value="numerical_range_test_node">Ranges</SelectItem>
                    <SelectItem value="numerical_test_node">Numerical</SelectItem>
                    <SelectItem value="categorical_test_node">Categorical</SelectItem>
                    <SelectItem value="leaf">Value</SelectItem>
                    </SelectContent>
                </Select>
                {
                selectedNode.data.node_type == 'numerical_test_node'?
                <NumericalConfig
                    node={selectedNode as Node<NumericalNodeData>}
                    updateNode={updateNode}
                /> : selectedNode.data.node_type == 'categorical_test_node'?
                <CategoricalConfig
                    node={selectedNode as Node<CategoricalNodeData>}
                    updateNode={updateNode}
                /> : selectedNode.data.node_type == 'numerical_range_test_node'?
                <RangeConfig
                    node={selectedNode as Node<RangeNodeData>}
                    updateNode={updateNode}
                /> : 
                <LeafConfig
                    node={selectedNode as Node<LeafNodeData>}
                    updateNode={updateNode}
                />
                }
                </div>

            </SidebarGroupContent>
            </SidebarGroup>
            </SidebarContent>
            </>)
    }

    return (
        <Sidebar side="right">
            {renderConfig()}
            <SidebarRail />
        </Sidebar>

    )
}