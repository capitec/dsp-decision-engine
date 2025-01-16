import React, { useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast"
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Dialog, DialogContent, DialogDescription, DialogHeader,
  DialogTitle, DialogTrigger
} from "@/components/ui/dialog";
import { TableEditor } from "./TableEditor";
import { ArrowUpDown, X, Plus, Table as TableIcon, Download } from "lucide-react";
import {
  SidebarGroup, SidebarGroupContent, SidebarGroupLabel
} from "@components/ui/sidebar";
import { 
  useEdges, useFeatures, useLeafOrder, useNodes, useProjectMetadata, 
  useTreeOutput
} from "@components/editor/EditorContext";
import { exportState, formatTree } from "@components/editor/util";
import FIleUploadButton from "./FileUpload"

export function ProjectSidebar() {
  const { toast } = useToast();
  const [nodes, setNodes] = useNodes();
  const [edges, setEdges] = useEdges();
  const [features, setFeatures] = useFeatures();
  const [metadata, setMetadata] = useProjectMetadata();
  const [treeOutput] = useTreeOutput();
  const [leafOrder] = useLeafOrder();

  const downloadLinkRef = useRef<HTMLAnchorElement | null>(null);

  const handleDownload = () => {
    if (!downloadLinkRef.current){
      toast({
        variant: "destructive",
        title: "Download Error",
        description: "Could not download the project due to issues with HTML"
      });
      return
    }
    let state;
    try {
      state = exportState(
        nodes, edges, features, leafOrder, metadata, treeOutput
      )
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Invalid Tree",
        description: (error as Error).message
      });
      return
    }
    const json = JSON.stringify(state, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    downloadLinkRef.current.href = url;
    downloadLinkRef.current.click();
  };

  const handleRearrange = useCallback(() => {
    formatTree(nodes, edges).then(newNodes => {
      setNodes(newNodes);
    });
  }, [nodes, edges, setNodes]);

  const handleAddFeature = useCallback(() => {
    setFeatures(prevFeatures => {
      const newFeature = `Feature ${prevFeatures.length + 1}`;
      if (prevFeatures.includes(newFeature)) {
        toast({
          variant: "destructive",
          title: "Feature exists",
          description: "A feature with this name already exists"
        });
        return prevFeatures;
      }
      return [...prevFeatures, newFeature];
    });
  }, [setFeatures, toast]);

  const handleDeleteFeature = useCallback((index: number) => {
    setFeatures(prevFeatures => {
      const nodeUsingFeature = nodes.find(node => 
        node.data.split_feature_id === index
      );

      if (nodeUsingFeature) {
        toast({
          variant: "destructive",
          title: "Cannot delete feature",
          description: `Feature is in use by node ${nodeUsingFeature.id}`
        });
        return prevFeatures;
      }

      const newFeatures = prevFeatures.filter((_, i) => i !== index);
      
      setNodes(prevNodes => 
        prevNodes.map(node => {
          if (!node.data.split_feature_id) return node;
          if (node.data.split_feature_id > index) {
            return {
              ...node,
              data: {
                ...node.data,
                split_feature_id: node.data.split_feature_id - 1
              }
            };
          }
          return node;
        })
      );

      return newFeatures;
    });
  }, [nodes, setNodes, setFeatures, toast]);

  const handleUpdateFeature = useCallback((index: number, value: string) => {
    setFeatures(prevFeatures => {
      const newFeatures = [...prevFeatures];
      newFeatures[index] = value;
      return newFeatures;
    });
  }, [setFeatures]);

  const handleMetadataChange = useCallback((field: keyof typeof metadata) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setMetadata(prev => ({
      ...prev,
      [field]: e.target.value
    }));
  }, [setMetadata]);

  // Rest of the JSX remains the same, but update event handlers:
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
              onChange={handleMetadataChange('name')}
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
              onChange={handleMetadataChange('description')}
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
          <div className="max-h-60 overflow-y-auto flex flex-col gap-2 mb-2">
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
          </div>
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
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="outline" className="w-full">
                <TableIcon className="h-4 w-4 mr-2" />
                Configure Outputs
              </Button>
            </DialogTrigger>
            <DialogContent className="md:max-w-[1000px] lg:max-w-[1500px] md:max-h-[600px] lg:max-h-[800px] overflow-auto flex flex-col">
              <DialogHeader>
                <DialogTitle>Edit Outputs</DialogTitle>
                <DialogDescription>
                  Make changes to the output elements of the tree.
                </DialogDescription>
              </DialogHeader>
              <TableEditor />
            </DialogContent>
          </Dialog>
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
                <p>Automatically format and organize the decision tree layout</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <FIleUploadButton/>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  onClick={handleDownload}
                  className="w-full"
                >
                  <a ref={downloadLinkRef} style={{ display: 'none' }} download="data.json"></a>
                  <Download className="h-4 w-4 mr-2" />
                  Download Project
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Download projectJson</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </SidebarGroupContent>
      </SidebarGroup>
    </>
  );
}

