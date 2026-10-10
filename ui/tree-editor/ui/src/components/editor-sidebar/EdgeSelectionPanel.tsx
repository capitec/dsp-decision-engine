import React, { useCallback, useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem } from "@/components/ui/command";
import { ChevronsUpDown, X } from "lucide-react";
import { useEditorStore } from '../editor/useEditorStore';
import { getNodeEdgeOptions } from '../../lib/node-utils';
import { cn } from "@/lib/utils";

interface EdgeSelectionPanelProps {
  edge: any;
}

export const EdgeSelectionPanel: React.FC<EdgeSelectionPanelProps> = ({ edge }) => {
  const nodes = useEditorStore((state) => state.nodes);
  const edges = useEditorStore((state) => state.edges);
  const updateEdge = useEditorStore((state) => state.updateEdge);
  const [open, setOpen] = useState(false);

  // Get the parent node of the selected edge
  const edgeParent = useMemo(() => {
    if (!edge) return null;
    return nodes.find(n => n.id === edge.source) || null;
  }, [edge, nodes]);

  // Generate available edge options based on parent node type
  // v2: variables are now inline in node data as ParameterReference objects
  const edgeOptions = useMemo(() => {
    if (!edgeParent) return [];
    return getNodeEdgeOptions(edgeParent.data, {});
  }, [edgeParent]);

  // Get current assignments - support both single number and array
  const currentAssignments = useMemo(() => {
    const sourceIndex = (edge?.data as any)?.sourceIndex;
    if (Array.isArray(sourceIndex)) {
      return sourceIndex;
    } else if (typeof sourceIndex === 'number') {
      return [sourceIndex];
    } else {
      return [0]; // default
    }
  }, [edge]);

  // Update edge assignment with multiple values
  const updateEdgeAssignment = useCallback((newAssignments: number[]) => {
    // For backward compatibility, if only one assignment, store as single number
    const assignmentValue = newAssignments.length === 1 ? newAssignments[0] : newAssignments;
    updateEdge(edge.id, { data: { ...edge.data, sourceIndex: assignmentValue } });
  }, [updateEdge, edge.id, edge.data]);

  // Toggle assignment
  const toggleAssignment = useCallback((optionValue: number) => {
    const newAssignments = currentAssignments.includes(optionValue)
      ? currentAssignments.filter(val => val !== optionValue)
      : [...currentAssignments, optionValue].sort((a, b) => a - b);
    
    updateEdgeAssignment(newAssignments);
  }, [currentAssignments, updateEdgeAssignment]);

  // Remove specific assignment
  const removeAssignment = useCallback((optionValue: number) => {
    const newAssignments = currentAssignments.filter(val => val !== optionValue);
    updateEdgeAssignment(newAssignments.length > 0 ? newAssignments : [0]); // Ensure at least one assignment
  }, [currentAssignments, updateEdgeAssignment]);

  if (!edge || !edgeParent) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Edge Properties</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-500">No edge selected or parent node not found.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Edge Properties</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <Label className="text-xs text-gray-500">Edge ID</Label>
          <p className="font-mono text-sm">{edge.id}</p>
        </div>

        <div>
          <Label className="text-xs text-gray-500">From Node</Label>
          <p className="text-sm">{edge.source}</p>
        </div>

        <div>
          <Label className="text-xs text-gray-500">To Node</Label>
          <p className="text-sm">{edge.target}</p>
        </div>

        {edgeOptions.length > 0 && (
          <div className="space-y-3">
            <Label>Branch Assignment</Label>
            
            {/* Multi-select dropdown */}
            <Popover open={open} onOpenChange={setOpen}>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  role="combobox"
                  aria-expanded={open}
                  className="w-full justify-between min-h-[2.5rem] h-auto"
                >
                  <div className="flex flex-wrap gap-1 flex-1 text-left">
                    {currentAssignments.length === 0 ? (
                      <span className="text-muted-foreground">Select branches...</span>
                    ) : (
                      currentAssignments.map(assignment => {
                        const option = edgeOptions.find(opt => opt.value === assignment);
                        return (
                          <Badge
                            key={assignment}
                            variant="secondary"
                            className="text-xs flex items-center gap-1"
                          >
                            {option?.label || `Branch ${assignment}`}
                            <X
                              data-testid="remove-branch"
                              className="h-3 w-3 cursor-pointer hover:text-destructive"
                              onClick={(e) => {
                                e.stopPropagation();
                                removeAssignment(assignment);
                              }}
                            />
                          </Badge>
                        );
                      })
                    )}
                  </div>
                  <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-full p-0" align="start">
                <Command>
                  <CommandInput placeholder="Search branches..." />
                  <CommandEmpty>No branches found.</CommandEmpty>
                  <CommandGroup className="max-h-64 overflow-auto">
                    {edgeOptions.map((option) => {
                      const isSelected = currentAssignments.includes(option.value);
                      return (
                        <CommandItem
                          key={option.value}
                          onSelect={() => toggleAssignment(option.value)}
                          className="cursor-pointer"
                        >
                          <div className="flex items-center gap-2 flex-1">
                            <Checkbox 
                              checked={isSelected}
                              className="data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                            />
                            <div className="flex-1">
                              <div className="font-medium">{option.label}</div>
                              <div className="text-xs text-muted-foreground">{option.description}</div>
                            </div>
                          </div>
                        </CommandItem>
                      );
                    })}
                  </CommandGroup>
                </Command>
              </PopoverContent>
            </Popover>

            {/* Show summary of current assignments */}
            {currentAssignments.length > 1 && (
              <div className="text-xs text-muted-foreground bg-muted/30 p-2 rounded">
                This edge will trigger for {currentAssignments.length} different branches: {
                  currentAssignments.map(assignment => {
                    const option = edgeOptions.find(opt => opt.value === assignment);
                    return option?.label || `Branch ${assignment}`;
                  }).join(', ')
                }
              </div>
            )}
          </div>
        )}

        <div className="pt-2 border-t">
          <p className="text-xs text-gray-500">
            This edge connects from <span className="font-mono">{edge.source}</span> to{' '}
            <span className="font-mono">{edge.target}</span> via branch{currentAssignments.length > 1 ? 'es' : ''} {currentAssignments.join(', ')}.
          </p>
        </div>
      </CardContent>
    </Card>
  );
};