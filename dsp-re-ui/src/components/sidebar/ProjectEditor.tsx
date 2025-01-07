// ConfigSidebar.tsx
import React, { useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { TableEditor } from "./TableEditor";
import { ArrowUpDown, X, Plus, Table as TableIcon } from "lucide-react";
import type { TreeOutput, ProjectMetadata } from "./types";
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
} from "@components/ui/sidebar";
import { useEdges, useFeatures, useNodes } from "@components/editor/EditorContext";
import { formatTree } from "@components/editor/util";

export function ProjectSidebar() {
  const [nodes, setNodes, onNodesChange] = useNodes();
  const [edges, setEdges, onEdgesChange] = useEdges();
  const [features, setFeatures] = useFeatures();

  const handleRearrange = useCallback(()=>{
    formatTree(nodes, edges).then(newNodes => {
        setNodes(newNodes)
    })
  }, [nodes, edges])

  const [metadata, setMetadata] = React.useState<ProjectMetadata>({
    name: "",
    description: "",
  });
  const [treeOutputs, setTreeOutputs] = React.useState<TreeOutput[]>([
    { column1: "", column2: "" },
  ]);

  const handleAddFeature = () => {
    setFeatures([...features, `Feature ${features.length + 1}`]);
  };

  const handleDeleteFeature = (index: number) => {
    setFeatures(features.filter((_, i) => i !== index));
  };

  const handleUpdateFeature = (index: number, value: string) => {
    const newFeatures = [...features];
    newFeatures[index] = value;
    setFeatures(newFeatures);
  };


  return (
    <>
      <SidebarGroup>
        <SidebarGroupLabel>Project Metadata</SidebarGroupLabel>
        <SidebarGroupContent>
          <div>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <label className="text-sm font-medium">Project Name</label>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Enter the name of your project</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <Input
              value={metadata.name}
              onChange={(e) =>
                setMetadata({ ...metadata, name: e.target.value })
              }
              placeholder="My Project"
              className="mt-1.5"
            />
          </div>

          <div>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <label className="text-sm font-medium">Description</label>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Briefly describe your project</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <Textarea
              value={metadata.description}
              onChange={(e) =>
                setMetadata({ ...metadata, description: e.target.value })
              }
              placeholder="Project description..."
              className="mt-1.5"
            />
          </div>
        </SidebarGroupContent>
      </SidebarGroup>

      {/* Features Section */}
      <SidebarGroup>
        <SidebarGroupLabel>Features</SidebarGroupLabel>
        <SidebarGroupContent>
          {features.map((feature, index) => (
            <div key={index} className="flex gap-2">
              <Input
                value={feature}
                onChange={(e) => handleUpdateFeature(index, e.target.value)}
                placeholder="Feature name"
              />
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDeleteFeature(index)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Delete feature</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          ))}

          <Button
            variant="outline"
            onClick={handleAddFeature}
            className="w-full"
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Feature
          </Button>
        </SidebarGroupContent>
      </SidebarGroup>

      {/* Outputs Section */}
      <SidebarGroup>
        <SidebarGroupLabel>Outputs</SidebarGroupLabel>
        <SidebarGroupContent>
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" className="w-full">
                <TableIcon className="h-4 w-4 mr-2" />
                Configure Outputs
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-[600px]">
              {<div>Hi i am an editor</div>}
              <TableEditor
                outputs={treeOutputs}
                onOutputsChange={setTreeOutputs}
              />
            </PopoverContent>
          </Popover>
        </SidebarGroupContent>
      </SidebarGroup>

      {/* Utilities Section */}
      <SidebarGroup>
        <SidebarGroupLabel>Utilities</SidebarGroupLabel>
        <SidebarGroupContent>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  onClick={handleRearrange}
                  className="w-full"
                >
                  <ArrowUpDown className="h-4 w-4 mr-2" />
                  Rearrange Decision Tree
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>
                  Automatically format and organize the decision tree layout
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </SidebarGroupContent>
      </SidebarGroup>
    </>
  );
}
