import React, { useContext, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Upload } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useToast } from "@/hooks/use-toast";
import { Loader2 } from "lucide-react";
import { loadState } from "@components/editor/util";
import { SourceData } from "@components/editor/types";
import { EditorContext } from "@components/editor/EditorContext";
import { DialogDescription } from "@radix-ui/react-dialog";

interface JsonData {
  [key: string]: unknown;
}

const FileUploadDialog = () => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [dialogOpen, setDialogOpen] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { toast } = useToast();
  const {setEdges, setNodes, setFeatures, setMetadata, setTreeOutput, setLeafOrder} = useContext(EditorContext);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file?.type === "application/json") {
      handleFileUpload(file);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const file = e.target.files?.[0];
    if (file) handleFileUpload(file);
  };

  const handleButtonClick = (): void => {
    fileInputRef.current?.click();
  };

  const handleFileUpload = (file: File): void => {
    const reader = new FileReader();
    setIsLoading(true);

    const readerPromise = new Promise<ProgressEvent<FileReader>>((resolve, reject) => {
        reader.onload = resolve;
        reader.onerror = reject;
    })
    readerPromise
        .then((evt) => JSON.parse(evt.target?.result as string) as SourceData)
        .then((data: SourceData)=>loadState(data, file.name))
        .then(({nodes, edges, features, leafOrder, metadata, treeOutput})=>{
            setEdges(edges)
            setNodes(nodes)
            setFeatures(features)
            setMetadata(metadata)
            setTreeOutput(treeOutput)
            setLeafOrder(leafOrder)
        }).then(()=>{
            toast({
                title: "Success",
                description: "JSON file loaded successfully",
            });
            setDialogOpen(false)
        })
        .catch(err=>{
            console.error("Error parsing JSON:", err);
            toast({
              variant: "destructive",
              title: "Error",
              description: "Failed to load JSON file",
            });
        }).finally(()=>{
            setIsLoading(false);
        })
    reader.readAsText(file);
  };

  return (
    <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <DialogTrigger asChild>
              <Button variant="outline" className="w-full">
                <Upload className="mr-2 h-4 w-4" />
                Load Project
              </Button>
            </DialogTrigger>
          </TooltipTrigger>
          <TooltipContent>
            <p>Upload new project</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Upload JSON File</DialogTitle>

            <DialogDescription>
                Choose a json file to download to begin editing
            </DialogDescription>
        </DialogHeader>
        {isLoading && (
          <div className="absolute inset-0 bg-background/80 flex items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        )}
        <div
          className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="application/json"
            onChange={handleFileInput}
            className="hidden"
            ref={fileInputRef}
            id="fileInput"
          />
          <label htmlFor="fileInput" className="cursor-pointer">
            <p>Drag and drop JSON file here</p>
            <p className="text-sm text-gray-500">or</p>
            <Button variant="secondary" size="sm" className="mt-2"  onClick={handleButtonClick}>
              Choose File
            </Button>
          </label>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default FileUploadDialog;