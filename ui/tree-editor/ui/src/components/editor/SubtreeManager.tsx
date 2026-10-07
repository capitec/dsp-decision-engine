import React, { useState, useCallback, useMemo } from 'react';
import { useReactFlow } from '@xyflow/react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  GripVertical,
  Edit3,
  Check,
  X,
  Navigation,
  Plus
} from "lucide-react";
import {
  DragDropContext,
  Droppable,
  Draggable,
  DroppableProvided,
  DraggableProvided,
  DropResult
} from '@hello-pangea/dnd';
import { useEditorStore } from './useEditorStore';
import {
  findRootNodes,
  reorderSubtrees
} from './util';

interface SubtreeManagerProps {}

export const SubtreeManager: React.FC<SubtreeManagerProps> = () => {
  const nodes = useEditorStore((state) => state.nodes);
  const edges = useEditorStore((state) => state.edges);
  const subtrees = useEditorStore((state) => state.subtrees);
  const setSubtrees = useEditorStore((state) => state.setSubtrees);
  const hasSubtreeErrors = useEditorStore((state) => state.hasSubtreeErrors);
  const { fitView, getNode } = useReactFlow();
  
  // Local UI state
  const [activeSubtreeId, setActiveSubtreeId] = useState<string | null>(null);
  const [editingSubtreeId, setEditingSubtreeId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState<string>('');

  // Handle navigation to subtree root node
  const handleNavigateToSubtree = useCallback((subtreeId: string) => {
    const subtree = subtrees.find(st => st.id === subtreeId);
    if (!subtree) return;

    const rootNode = getNode(subtree.rootNodeId);
    if (!rootNode) return;

    // Center the view on the root node with animation
    fitView({
      nodes: [rootNode],
      duration: 800,
      padding: 0.5,
    });

    // Set as active subtree
    setActiveSubtreeId(subtreeId);

    // Update subtrees to mark this one as active
    const updatedSubtrees = subtrees.map(st => ({
      ...st,
      isActive: st.id === subtreeId
    }));
    setSubtrees(updatedSubtrees);
  }, [subtrees, getNode, fitView, setSubtrees]);

  // Get root nodes that aren't yet subtrees and filter out hidden subtrees
  const rootNodes = useMemo(() => findRootNodes(nodes, edges), [nodes, edges]);
  const visibleSubtrees = useMemo(() => subtrees.filter(st => !st.hidden), [subtrees]);
  const hasUnmanagedRoots = rootNodes.length > visibleSubtrees.length;

  // Handle drag and drop reordering
  const handleDragEnd = useCallback((result: DropResult) => {
    if (!result.destination) return;

    const fromIndex = result.source.index;
    const toIndex = result.destination.index;

    if (fromIndex === toIndex) return;

    // Reorder only visible subtrees, then merge back with hidden ones
    const reorderedVisible = reorderSubtrees(visibleSubtrees, fromIndex, toIndex);
    const hiddenSubtrees = subtrees.filter(st => st.hidden);

    // Combine visible and hidden subtrees
    const allSubtrees = [...reorderedVisible, ...hiddenSubtrees];
    setSubtrees(allSubtrees);
  }, [visibleSubtrees, subtrees, setSubtrees]);

  // Start editing subtree name
  const startEditing = useCallback((subtreeId: string, currentName: string) => {
    setEditingSubtreeId(subtreeId);
    setEditingName(currentName);
  }, []);

  // Save edited name
  const saveEditedName = useCallback(() => {
    if (!editingSubtreeId || editingName.trim() === '') return;

    const trimmedName = editingName.trim().slice(0, 40); // Max 40 chars

    const updatedSubtrees = subtrees.map(st =>
      st.id === editingSubtreeId
        ? { ...st, name: trimmedName }
        : st
    );
    setSubtrees(updatedSubtrees);

    setEditingSubtreeId(null);
    setEditingName('');
  }, [editingSubtreeId, editingName, subtrees, setSubtrees]);

  // Cancel editing
  const cancelEditing = useCallback(() => {
    setEditingSubtreeId(null);
    setEditingName('');
  }, []);

  // Handle adding a new disconnected subtree (creates a leaf node in center)
  const handleAddSubtree = useCallback(() => {
    // This would trigger the parent component's handleAddNode function
    // For now, we'll just show a message as this requires integration with the parent
    console.log('Add subtree functionality requires parent integration');
  }, []);

  if (visibleSubtrees.length === 0 && rootNodes.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Subtrees</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-gray-500 mb-3">
            No subtrees detected. Create disconnected trees to manage execution priority.
          </p>
          <Button 
            onClick={handleAddSubtree}
            variant="outline" 
            size="sm"
            className="w-full"
          >
            <Plus className="h-3 w-3 mr-1" />
            Add Subtree
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={hasSubtreeErrors ? 'ring-2 ring-red-500' : ''}>
      <CardHeader>
        <CardTitle className="text-sm flex items-center justify-between">
          <span className="flex items-center gap-2">
            Subtrees
            {hasSubtreeErrors && (
              <span className="text-red-500 text-xs">⚠ Has errors</span>
            )}
          </span>
          <Badge variant="secondary" className="text-xs">
            {visibleSubtrees.length}
          </Badge>
        </CardTitle>
        <p className="text-xs text-gray-500">
          Drag to reorder priority (top = highest)
        </p>
      </CardHeader>
      <CardContent className="space-y-2">
        <DragDropContext onDragEnd={handleDragEnd}>
          <Droppable droppableId="subtrees">
            {(provided: DroppableProvided) => (
              <div
                {...provided.droppableProps}
                ref={provided.innerRef}
                className="space-y-1"
              >
                {visibleSubtrees.map((subtree, index) => (
                  <Draggable 
                    key={subtree.id} 
                    draggableId={subtree.id} 
                    index={index}
                  >
                    {(provided: DraggableProvided, snapshot) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        className={`
                          flex items-center space-x-2 p-2 rounded border
                          ${subtree.isActive 
                            ? 'bg-blue-50 border-blue-200' 
                            : 'bg-gray-50 border-gray-200'
                          }
                          ${snapshot.isDragging ? 'shadow-lg' : ''}
                          hover:bg-gray-100 transition-colors
                        `}
                      >
                        {/* Drag Handle */}
                        <div
                          {...provided.dragHandleProps}
                          className="text-gray-400 hover:text-gray-600 cursor-grab active:cursor-grabbing"
                        >
                          <GripVertical className="h-3 w-3" />
                        </div>

                        {/* Priority Badge */}
                        <Badge 
                          variant= "default"
                          className="text-xs px-1 py-0 min-w-[20px] justify-center"
                        >
                          {index + 1}
                        </Badge>

                        {/* Name (editable or clickable) */}
                        <div className="flex-1 min-w-0">
                          {editingSubtreeId === subtree.id ? (
                            <div className="flex items-center space-x-1">
                              <Input
                                value={editingName}
                                onChange={(e) => setEditingName(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') saveEditedName();
                                  if (e.key === 'Escape') cancelEditing();
                                }}
                                className="h-6 text-xs px-1"
                                placeholder="Subtree name"
                                maxLength={40}
                                autoFocus
                              />
                              <Button
                                onClick={saveEditedName}
                                size="sm"
                                variant="ghost"
                                className="h-5 w-5 p-0"
                              >
                                <Check className="h-3 w-3 text-green-600" />
                              </Button>
                              <Button
                                onClick={cancelEditing}
                                size="sm"
                                variant="ghost"
                                className="h-5 w-5 p-0"
                              >
                                <X className="h-3 w-3 text-red-600" />
                              </Button>
                            </div>
                          ) : (
                            <div className="flex items-center space-x-1 group">
                              <button
                                onClick={() => handleNavigateToSubtree(subtree.id)}
                                className="text-xs font-medium truncate hover:text-blue-600 transition-colors cursor-pointer flex-1 text-left"
                                title={subtree.name}
                              >
                                {subtree.name}
                              </button>
                              <Button
                                onClick={() => startEditing(subtree.id, subtree.name)}
                                size="sm"
                                variant="ghost"
                                className="h-4 w-4 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                              >
                                <Edit3 className="h-3 w-3 text-gray-400" />
                              </Button>
                            </div>
                          )}
                        </div>

                        {/* Navigation Button */}
                        {editingSubtreeId !== subtree.id && (
                          <Button
                            onClick={() => handleNavigateToSubtree(subtree.id)}
                            size="sm"
                            variant="ghost"
                            className="h-6 w-6 p-0"
                            title="Navigate to root node"
                          >
                            <Navigation className="h-3 w-3 text-gray-500" />
                          </Button>
                        )}
                      </div>
                    )}
                  </Draggable>
                ))}
                {provided.placeholder}
              </div>
            )}
          </Droppable>
        </DragDropContext>

        {/* Add Subtree Button */}
        {hasUnmanagedRoots && (
          <div className="pt-2 border-t">
            <Button 
              onClick={handleAddSubtree}
              variant="outline" 
              size="sm"
              className="w-full"
            >
              <Plus className="h-3 w-3 mr-1" />
              Add Subtree
            </Button>
          </div>
        )}

        {/* Help Text */}
        <div className="pt-2 border-t text-xs text-gray-500 space-y-1">
          <p>• Click names to navigate to roots</p>
          <p>• Higher position = higher priority</p>
          <p>• All subtrees run concurrently</p>
        </div>
      </CardContent>
    </Card>
  );
};