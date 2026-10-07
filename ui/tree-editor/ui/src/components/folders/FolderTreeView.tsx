import React, { useState, useMemo } from 'react';
import {
  ChevronRight,
  ChevronDown,
  FolderIcon,
  FolderOpen,
  MoreVertical,
  Edit2,
  Trash2,
  Share,
  Move,
  Plus,
  Users,
  Lock,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { cn } from '@/lib/utils';
import type { Folder, AccessMode } from '@/types/folder-types';

interface FolderTreeViewProps {
  folders: Folder[];
  currentFolderId: string | null;
  onFolderSelect: (folderId: string | null) => void;
  onCreateFolder?: (parentId?: string) => void;
  onRenameFolder?: (folder: Folder) => void;
  onDeleteFolder?: (folder: Folder) => void;
  onShareFolder?: (folder: Folder) => void;
  onMoveFolder?: (folder: Folder) => void;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  className?: string;
}

interface FolderTreeNode extends Folder {
  children: FolderTreeNode[];
  level: number;
}

export const FolderTreeView: React.FC<FolderTreeViewProps> = ({
  folders,
  currentFolderId,
  onFolderSelect,
  onCreateFolder,
  onRenameFolder,
  onDeleteFolder,
  onShareFolder,
  onMoveFolder,
  loading = false,
  error = null,
  onRetry,
  className,
}) => {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());

  // Build tree structure
  const folderTree = useMemo(() => {
    const buildTree = (parentId: string | null | undefined, level: number = 0): FolderTreeNode[] => {
      return folders
        .filter(f => {
          // Handle both null and undefined as "no parent"
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

  const renderAccessBadge = (accessMode: AccessMode, isShared: boolean) => {
    if (accessMode === 'read') {
      return (
        <div className="flex items-center gap-1 px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-xs" title="Read-only access">
          <Lock className="h-3 w-3" />
        </div>
      );
    }

    if (isShared) {
      return (
        <div className="flex items-center gap-1 px-1.5 py-0.5 bg-purple-100 text-purple-700 rounded text-xs" title="Shared folder">
          <Users className="h-3 w-3" />
        </div>
      );
    }

    return null;
  };

  const renderFolderNode = (node: FolderTreeNode) => {
    const isExpanded = expandedFolders.has(node.id);
    const isSelected = currentFolderId === node.id;
    const hasChildren = node.children.length > 0;
    const canEdit = node.access_mode === 'write' || node.access_mode === 'admin';
    const canDelete = node.access_mode === 'admin';

    return (
      <div key={node.id}>
        <div
          className={cn(
            'group flex items-center gap-1 py-1.5 px-2 rounded-md hover:bg-gray-100 cursor-pointer transition-colors',
            isSelected && 'bg-blue-50 hover:bg-blue-100',
            'relative'
          )}
          style={{ paddingLeft: `${node.level * 16 + 8}px` }}
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
            <div className="w-5" /> // Spacer for alignment
          )}

          {/* Folder icon */}
          <div className="flex-shrink-0">
            {isExpanded ? (
              <FolderOpen className="h-4 w-4 text-blue-500" />
            ) : (
              <FolderIcon className="h-4 w-4 text-gray-500" />
            )}
          </div>

          {/* Folder name */}
          <button
            onClick={() => onFolderSelect(node.id)}
            className={cn(
              'flex-1 text-left text-sm truncate',
              isSelected ? 'font-medium text-blue-700' : 'text-gray-700'
            )}
            title={node.name}
          >
            {node.name}
          </button>

          {/* Access badge */}
          {renderAccessBadge(node.access_mode, node.is_shared || false)}

          {/* Project count */}
          {(node.project_count || 0) > 0 && (
            <span className="text-xs text-gray-500 flex-shrink-0" title={`${node.project_count} projects`}>
              {node.project_count}
            </span>
          )}

          {/* Actions menu */}
          <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
            <DropdownMenu>
              <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                  <MoreVertical className="h-3.5 w-3.5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={() => onCreateFolder?.(node.id)} disabled={!canEdit}>
                  <Plus className="mr-2 h-4 w-4" />
                  Create Subfolder
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => onRenameFolder?.(node)} disabled={!canEdit}>
                  <Edit2 className="mr-2 h-4 w-4" />
                  Rename
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => onShareFolder?.(node)}>
                  <Share className="mr-2 h-4 w-4" />
                  Share
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => onMoveFolder?.(node)} disabled={!canEdit}>
                  <Move className="mr-2 h-4 w-4" />
                  Move
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={() => onDeleteFolder?.(node)}
                  disabled={!canDelete}
                  className="text-red-600 focus:text-red-600"
                >
                  <Trash2 className="mr-2 h-4 w-4" />
                  Delete
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        {/* Render children if expanded */}
        {isExpanded && hasChildren && (
          <div>{node.children.map(child => renderFolderNode(child))}</div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className={cn('space-y-2 p-2', className)}>
        {[1, 2, 3].map(i => (
          <div key={i} className="flex items-center gap-2 px-2 py-1.5 animate-pulse">
            <div className="h-4 w-4 bg-gray-200 rounded" />
            <div className="h-4 flex-1 bg-gray-200 rounded" />
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn('p-4', className)}>
        <div className="text-center space-y-3">
          <div className="text-red-600 text-sm font-medium">Failed to load folders</div>
          <p className="text-xs text-gray-600">{error}</p>
          {onRetry && (
            <Button variant="outline" size="sm" onClick={onRetry}>
              Try Again
            </Button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={cn('space-y-1 p-2', className)}>
      {/* Root folder item */}
      <div
        className={cn(
          'flex items-center gap-2 py-1.5 px-2 rounded-md hover:bg-gray-100 cursor-pointer transition-colors',
          currentFolderId === null && 'bg-blue-50 hover:bg-blue-100'
        )}
        onClick={() => onFolderSelect(null)}
      >
        <FolderIcon className="h-4 w-4 text-gray-500" />
        <span
          className={cn(
            'text-sm flex-1',
            currentFolderId === null ? 'font-medium text-blue-700' : 'text-gray-700'
          )}
        >
          All Projects
        </span>
      </div>

      {/* Folder tree */}
      {folderTree.length === 0 ? (
        <div className="px-2 py-8 text-center text-sm text-gray-500">
          <FolderIcon className="h-8 w-8 mx-auto mb-2 text-gray-400" />
          <p>No folders yet</p>
          {onCreateFolder && (
            <Button variant="link" size="sm" onClick={() => onCreateFolder()} className="mt-2">
              Create your first folder
            </Button>
          )}
        </div>
      ) : (
        <div>{folderTree.map(node => renderFolderNode(node))}</div>
      )}
    </div>
  );
};
