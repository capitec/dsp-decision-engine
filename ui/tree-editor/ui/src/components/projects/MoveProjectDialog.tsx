import React, { useState } from 'react';
import { FolderIcon, ChevronRight, ChevronDown } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import type { Folder } from '@/types/folder-types';
import type { ProjectResponse } from '@/api/types.gen';

interface MoveProjectDialogProps {
  open: boolean;
  project: ProjectResponse | null;
  folders: Folder[];
  currentFolderId: string | null;
  onClose: () => void;
  onConfirm: (projectId: string, newFolderId: string | null) => Promise<void>;
}

interface FolderTreeNode extends Folder {
  children: FolderTreeNode[];
  level: number;
}

export const MoveProjectDialog: React.FC<MoveProjectDialogProps> = ({
  open,
  project,
  folders,
  currentFolderId,
  onClose,
  onConfirm,
}) => {
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(currentFolderId);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);

  // Build folder tree
  const folderTree = React.useMemo(() => {
    const buildTree = (parentId: string | null | undefined, level: number = 0): FolderTreeNode[] => {
      return folders
        .filter(f => {
          const folderParentId = f.parent_id ?? null;
          const targetParentId = parentId ?? null;
          return folderParentId === targetParentId;
        })
        .map(folder => ({
          ...folder,
          level,
          children: buildTree(folder.id, level + 1),
        }))
        .sort((a, b) => a.name.localeCompare(b.name));
    };

    return buildTree(null);
  }, [folders]);

  const toggleExpanded = (folderId: string) => {
    setExpandedFolders(prev => {
      const newSet = new Set(prev);
      if (newSet.has(folderId)) {
        newSet.delete(folderId);
      } else {
        newSet.add(folderId);
      }
      return newSet;
    });
  };

  const handleConfirm = async () => {
    if (!project) return;

    try {
      setLoading(true);
      await onConfirm(project.id, selectedFolderId);
      onClose();
    } catch (error) {
      console.error('Failed to move project:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderFolderNode = (node: FolderTreeNode) => {
    const isExpanded = expandedFolders.has(node.id);
    const isSelected = selectedFolderId === node.id;
    const hasChildren = node.children.length > 0;
    const isCurrentLocation = currentFolderId === node.id;

    return (
      <div key={node.id}>
        <button
          onClick={() => setSelectedFolderId(node.id)}
          className={cn(
            'w-full flex items-center gap-1 py-2 px-2 rounded-md hover:bg-gray-100 cursor-pointer transition-colors text-left',
            isSelected && 'bg-blue-50 hover:bg-blue-100',
            isCurrentLocation && 'opacity-50 cursor-not-allowed'
          )}
          style={{ paddingLeft: `${node.level * 16 + 8}px` }}
          disabled={isCurrentLocation}
        >
          {/* Expand/collapse button */}
          {hasChildren ? (
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleExpanded(node.id);
              }}
              className="flex-shrink-0 p-0.5 hover:bg-gray-200 rounded"
              aria-label={isExpanded ? 'Collapse folder' : 'Expand folder'}
            >
              {isExpanded ? (
                <ChevronDown className="h-4 w-4 text-gray-600" />
              ) : (
                <ChevronRight className="h-4 w-4 text-gray-600" />
              )}
            </button>
          ) : (
            <div className="w-5" />
          )}

          {/* Folder icon */}
          <FolderIcon className={cn(
            "h-4 w-4 flex-shrink-0",
            isSelected ? "text-blue-500" : "text-gray-500"
          )} />

          {/* Folder name */}
          <span className={cn(
            'flex-1 text-sm truncate',
            isSelected ? 'font-medium text-blue-700' : 'text-gray-700',
            isCurrentLocation && 'text-gray-400'
          )}>
            {node.name}
            {isCurrentLocation && ' (current)'}
          </span>
        </button>

        {/* Render children if expanded */}
        {isExpanded && hasChildren && (
          <div>{node.children.map(child => renderFolderNode(child))}</div>
        )}
      </div>
    );
  };

  if (!project) return null;

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-md max-h-[600px] flex flex-col">
        <DialogHeader>
          <DialogTitle>Move Project</DialogTitle>
          <DialogDescription>
            Choose a destination folder for "{project.name}"
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto border rounded-md">
          {/* Root folder option */}
          <button
            onClick={() => setSelectedFolderId(null)}
            className={cn(
              'w-full flex items-center gap-2 py-2 px-3 hover:bg-gray-100 cursor-pointer transition-colors text-left border-b',
              selectedFolderId === null && 'bg-blue-50 hover:bg-blue-100',
              currentFolderId === null && 'opacity-50 cursor-not-allowed'
            )}
            disabled={currentFolderId === null}
          >
            <FolderIcon className={cn(
              "h-4 w-4",
              selectedFolderId === null ? "text-blue-500" : "text-gray-500"
            )} />
            <span className={cn(
              'text-sm flex-1',
              selectedFolderId === null ? 'font-medium text-blue-700' : 'text-gray-700',
              currentFolderId === null && 'text-gray-400'
            )}>
              Root (No folder)
              {currentFolderId === null && ' (current)'}
            </span>
          </button>

          {/* Folder tree */}
          <div className="p-2">
            {folderTree.length === 0 ? (
              <div className="py-8 text-center text-sm text-gray-500">
                <p>No folders available</p>
              </div>
            ) : (
              <div>{folderTree.map(node => renderFolderNode(node))}</div>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={loading}>
            Cancel
          </Button>
          <Button
            onClick={handleConfirm}
            disabled={loading || selectedFolderId === currentFolderId}
          >
            {loading ? 'Moving...' : 'Move Project'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
