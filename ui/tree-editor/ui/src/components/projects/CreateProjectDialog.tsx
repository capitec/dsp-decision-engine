import React, { useState, useCallback, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/hooks/use-toast";
import { createProjectApiV1ProjectsPost } from "@/api";
import { apiErrorMessage } from "@/lib/utils";
import type { ProjectType } from "@/api/types.gen";

interface CreateProjectDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onProjectCreated: () => void;
  projectType: ProjectType;
}

export const CreateProjectDialog: React.FC<CreateProjectDialogProps> = ({
  open,
  onOpenChange,
  onProjectCreated,
  projectType
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [creating, setCreating] = useState(false);
  const [errors, setErrors] = useState<{ name?: string; description?: string }>({});

  const validateForm = useCallback(() => {
    const newErrors: { name?: string; description?: string } = {};
    
    if (!name.trim()) {
      newErrors.name = 'Project name is required';
    } else if (name.trim().length < 2) {
      newErrors.name = 'Project name must be at least 2 characters';
    } else if (name.trim().length > 100) {
      newErrors.name = 'Project name must be less than 100 characters';
    }
    
    if (description && description.length > 500) {
      newErrors.description = 'Description must be less than 500 characters';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [name, description]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    try {
      setCreating(true);
      
      await createProjectApiV1ProjectsPost({
        body: {
          name: name.trim(),
          description: description.trim() || undefined,
          project_type: projectType
        }
      });
      
      toast({
        title: "Project created",
        description: `Project "${name.trim()}" has been created successfully.`,
      });
      
      // Reset form
      setName('');
      setDescription('');
      setErrors({});
      
      // Close dialog and refresh project list
      onOpenChange(false);
      onProjectCreated();
      
    } catch (error: any) {
      console.error('Failed to create project:', error);
      toast({
        title: "Creation failed",
        description: apiErrorMessage(error, "Failed to create project. Please try again."),
        variant: "destructive",
      });
    } finally {
      setCreating(false);
    }
  };

  const handleCancel = () => {
    if (!creating) {
      setName('');
      setDescription('');
      setErrors({});
      onOpenChange(false);
    }
  };

  // Auto-validate on input change
  React.useEffect(() => {
    if (name || description) {
      validateForm();
    }
  }, [name, description, validateForm]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Create New {projectType === 'tree' ? 'Tree' : 'Flat Rule'} Project</DialogTitle>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="project-name">Project Name *</Label>
            <Input
              id="project-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter project name"
              disabled={creating}
              autoFocus
              className={errors.name ? 'border-red-500' : ''}
            />
            {errors.name && (
              <p className="text-sm text-red-600">{errors.name}</p>
            )}
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="project-description">Description</Label>
            <Textarea
              id="project-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter project description (optional)"
              disabled={creating}
              rows={3}
              className={errors.description ? 'border-red-500' : ''}
            />
            {errors.description && (
              <p className="text-sm text-red-600">{errors.description}</p>
            )}
            <p className="text-xs text-gray-500">
              {description.length}/500 characters
            </p>
          </div>
        </form>
        
        <DialogFooter>
          <Button
            variant="outline"
            onClick={handleCancel}
            disabled={creating}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={creating || !name.trim()}
          >
            {creating ? 'Creating...' : 'Create Project'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};