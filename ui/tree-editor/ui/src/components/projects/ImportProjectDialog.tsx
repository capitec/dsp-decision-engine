import React, { useState, useRef, useCallback } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Upload, FileText, AlertCircle, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface ImportResult {
  project_id: string;
  project_name: string;
  project_type: 'tree' | 'flat_rule';
  validation_errors: unknown[];
}

interface ImportProjectDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onProjectImported: (projectId: string, projectType: 'tree' | 'flat_rule') => void;
}

export const ImportProjectDialog: React.FC<ImportProjectDialogProps> = ({
  open,
  onOpenChange,
  onProjectImported,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [projectName, setProjectName] = useState('');
  const [projectDescription, setProjectDescription] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const resetForm = () => {
    setSelectedFile(null);
    setProjectName('');
    setProjectDescription('');
    setError(null);
    setIsUploading(false);
    setIsDragOver(false);
  };

  const handleFileSelect = useCallback((file: File) => {
    if (!file.name.endsWith('.json')) {
      setError('Only JSON files are supported');
      return;
    }
    
    setSelectedFile(file);
    setError(null);
    
    // Auto-populate project name from filename if not set
    if (!projectName) {
      const nameFromFile = file.name.replace('.json', '').replace(/[_-]/g, ' ');
      setProjectName(nameFromFile);
    }
  }, [projectName]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const removeFile = () => {
    setSelectedFile(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedFile) {
      setError('Please select a JSON file to import');
      return;
    }

    if (!projectName.trim()) {
      setError('Project name is required');
      return;
    }

    try {
      setIsUploading(true);
      setError(null);
  
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('name', projectName);
      formData.append('description', projectDescription);

      const response = await fetch('/api/v1/projects/import', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        setError(errorData.detail || 'Import failed');
        return;
      }

      const result: ImportResult = await response.json();

      // Navigate to the project immediately based on its type
      onProjectImported(result.project_id, result.project_type);
      onOpenChange(false);
      resetForm();

    } catch (err: any) {
      console.error('Import failed:', err);
      setError(err.message || 'Failed to import project');
      } finally {
      setIsUploading(false);
    }
  };

  const handleDialogChange = (newOpen: boolean) => {
    if (!newOpen) {
      resetForm();
    }
    onOpenChange(newOpen);
  };

  return (
    <Dialog open={open} onOpenChange={handleDialogChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Import Project</DialogTitle>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* File Upload Area */}
          <div className="space-y-2">
            <Label>Project File</Label>
            <div
              className={cn(
                "border-2 border-dashed rounded-lg p-8 text-center transition-colors",
                isDragOver ? "border-blue-500 bg-blue-50" : "border-gray-300",
                selectedFile ? "border-green-500 bg-green-50" : ""
              )}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {selectedFile ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-center space-x-2">
                    <FileText className="h-8 w-8 text-green-600" />
                    <div className="text-left">
                      <p className="font-medium text-green-800">{selectedFile.name}</p>
                      <p className="text-sm text-green-600">
                        {(selectedFile.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={removeFile}
                      className="ml-2"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="h-12 w-12 mx-auto text-gray-400" />
                  <div>
                    <p className="text-lg font-medium text-gray-900">
                      Drop your JSON file here
                    </p>
                    <p className="text-sm text-gray-500">or</p>
                    <Button
                      type="button"
                      variant="outline"
                      onClick={handleBrowseClick}
                      className="mt-2"
                    >
                      Browse Files
                    </Button>
                  </div>
                  <p className="text-xs text-gray-500">
                    Supports JSON files only
                  </p>
                </div>
              )}
            </div>
            
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleFileInputChange}
              className="hidden"
            />
          </div>

          {/* Project Details */}
          <div className="space-y-4">
            <div>
              <Label htmlFor="name">Project Name</Label>
              <Input
                id="name"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                placeholder="Enter project name"
                required
                disabled={isUploading}
              />
            </div>
            
            <div>
              <Label htmlFor="description">Description (Optional)</Label>
              <Input
                id="description"
                value={projectDescription}
                onChange={(e) => setProjectDescription(e.target.value)}
                placeholder="Enter project description"
                disabled={isUploading}
              />
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="flex items-center space-x-2 p-3 bg-red-50 border border-red-200 rounded-md">
              <AlertCircle className="h-5 w-5 text-red-500" />
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex justify-end space-x-2 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={() => handleDialogChange(false)}
              disabled={isUploading}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={!selectedFile || isUploading}
            >
              {isUploading ? 'Importing...' : 'Import Project'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};