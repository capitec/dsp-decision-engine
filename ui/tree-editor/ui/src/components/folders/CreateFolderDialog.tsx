import React, { useState, useCallback, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from '@/hooks/use-toast';
import { useFolders } from '@/contexts/FolderContext';
import { FolderIcon, Home } from 'lucide-react';
import type { Folder } from '@/types/folder-types';

interface CreateFolderDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onFolderCreated?: (folder: Folder) => void;
  defaultParentId?: string;
}

export const CreateFolderDialog: React.FC<CreateFolderDialogProps> = ({
  open,
  onOpenChange,
  onFolderCreated,
  defaultParentId,
}) => {
  const { createFolder, folders, refreshFolders } = useFolders();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [parentId, setParentId] = useState<string | undefined>(defaultParentId);
  const [creating, setCreating] = useState(false);
  const [errors, setErrors] = useState<{ name?: string; description?: string }>({});

  // Update parent ID when default changes
  useEffect(() => {
    setParentId(defaultParentId);
  }, [defaultParentId]);

  const validateForm = useCallback(() => {
    const newErrors: { name?: string; description?: string } = {};

    if (!name.trim()) {
      newErrors.name = 'Folder name is required';
    } else if (name.trim().length < 2) {
      newErrors.name = 'Folder name must be at least 2 characters';
    } else if (name.trim().length > 255) {
      newErrors.name = 'Folder name must be less than 255 characters';
    }

    if (description && description.length > 500) {
      newErrors.description = 'Description must be less than 500 characters';
    }

    // Check for duplicate names in the same parent
    const siblings = folders.filter(f => f.parent_id === parentId);
    const duplicate = siblings.find(f => f.name.toLowerCase() === name.trim().toLowerCase());
    if (duplicate) {
      newErrors.name = 'A folder with this name already exists in this location';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [name, description, parentId, folders]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    try {
      setCreating(true);

      const newFolder = await createFolder({
        name: name.trim(),
        description: description.trim() || undefined,
        parent_id: parentId,
      });

      toast({
        title: 'Folder created',
        description: `Folder "${name.trim()}" has been created successfully.`,
      });

      // Reset form
      setName('');
      setDescription('');
      setParentId(defaultParentId);
      setErrors({});

      // Refresh folders and notify parent
      await refreshFolders();
      onFolderCreated?.(newFolder);

      // Close dialog
      onOpenChange(false);
    } catch (error: any) {
      console.error('Failed to create folder:', error);
      toast({
        title: 'Creation failed',
        description: error.message || 'Failed to create folder. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setCreating(false);
    }
  };

  const handleCancel = () => {
    if (!creating) {
      setName('');
      setDescription('');
      setParentId(defaultParentId);
      setErrors({});
      onOpenChange(false);
    }
  };

  // Auto-validate on input change
  useEffect(() => {
    if (name || description) {
      validateForm();
    }
  }, [name, description, validateForm]);

  // Get parent folder name for display
  const parentFolder = parentId ? folders.find(f => f.id === parentId) : null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Create New Folder</DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="folder-name">Folder Name *</Label>
            <Input
              id="folder-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter folder name"
              disabled={creating}
              autoFocus
              className={errors.name ? 'border-red-500' : ''}
            />
            {errors.name && <p className="text-sm text-red-600">{errors.name}</p>}
          </div>

          <div className="space-y-2">
            <Label htmlFor="folder-description">Description</Label>
            <Textarea
              id="folder-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter folder description (optional)"
              disabled={creating}
              rows={3}
              className={errors.description ? 'border-red-500' : ''}
            />
            {errors.description && <p className="text-sm text-red-600">{errors.description}</p>}
            <p className="text-xs text-gray-500">{description.length}/500 characters</p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="parent-folder">Parent Folder</Label>
            <Select value={parentId || 'root'} onValueChange={(val) => setParentId(val === 'root' ? undefined : val)}>
              <SelectTrigger id="parent-folder" disabled={creating}>
                <SelectValue placeholder="Select parent folder" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="root">
                  <div className="flex items-center gap-2">
                    <Home className="h-4 w-4" />
                    <span>Root (No parent)</span>
                  </div>
                </SelectItem>
                {folders
                  .filter(f => !f.parent_id) // Only show root-level folders for simplicity
                  .map(folder => (
                    <SelectItem key={folder.id} value={folder.id}>
                      <div className="flex items-center gap-2">
                        <FolderIcon className="h-4 w-4" />
                        <span>{folder.name}</span>
                      </div>
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
            {parentFolder && (
              <p className="text-xs text-gray-600">
                This folder will be created inside "{parentFolder.name}"
              </p>
            )}
          </div>
        </form>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel} disabled={creating}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={creating || !name.trim()}>
            {creating ? 'Creating...' : 'Create Folder'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
