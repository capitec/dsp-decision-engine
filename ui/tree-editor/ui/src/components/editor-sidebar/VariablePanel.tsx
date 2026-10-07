import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Alert,
  AlertDescription,
} from "@/components/ui/alert";
import { Plus, Trash2, Edit2, AlertCircle } from "lucide-react";
import { useEditorStore } from '../editor/useEditorStore';
import { extractParameterReferences, renameParameterInNodes, deleteParameterFromNodes, countParameterUsages } from '../editor/VariableActions';

interface VariablePanelProps {
  className?: string;
}

/**
 * Variable management panel for v2 format.
 *
 * Variables are inline ParameterReference objects in nodes.
 * This panel extracts variables from nodes and provides rename/delete.
 */
export const VariablePanel: React.FC<VariablePanelProps> = ({ className = "" }) => {
  const nodes = useEditorStore((state) => state.nodes);
  const setNodes = useEditorStore((state) => state.setNodes);

  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogMode, setDialogMode] = useState<'create' | 'rename'>('create');
  const [editingVariable, setEditingVariable] = useState<string | null>(null);
  const [variableName, setVariableName] = useState('');
  const [errors, setErrors] = useState<string[]>([]);

  // Extract all variable names from nodes
  const variables = useMemo(() => extractParameterReferences(nodes), [nodes]);

  // Find where each variable is used
  const variableUsage = useMemo(() => {
    const usage: Record<string, number> = {};
    variables.forEach(varName => {
      usage[varName] = countParameterUsages(nodes, varName);
    });
    return usage;
  }, [nodes, variables]);

  const openCreateDialog = () => {
    setDialogMode('create');
    setEditingVariable(null);
    setVariableName('');
    setErrors([]);
    setDialogOpen(true);
  };

  const openRenameDialog = (varName: string) => {
    setDialogMode('rename');
    setEditingVariable(varName);
    setVariableName(varName);
    setErrors([]);
    setDialogOpen(true);
  };

  const validateAndSave = () => {
    const newErrors: string[] = [];

    // Validate name
    const trimmedName = variableName.trim();
    if (!trimmedName) {
      newErrors.push('Variable name is required');
    }
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(trimmedName)) {
      newErrors.push('Variable name must be a valid identifier (letters, numbers, underscores; cannot start with a number)');
    }

    // Check for duplicate (only when creating or renaming to different name)
    if (trimmedName !== editingVariable && variables.includes(trimmedName)) {
      newErrors.push(`Variable "${trimmedName}" already exists`);
    }

    if (newErrors.length > 0) {
      setErrors(newErrors);
      return;
    }

    if (dialogMode === 'rename' && editingVariable) {
      // Rename across all nodes
      const updatedNodes = renameParameterInNodes(nodes, editingVariable, trimmedName);
      setNodes(updatedNodes);
    }
    // Note: For 'create', we don't actually create anything - variables are created
    // when used in nodes. This dialog just validates the name.

    setDialogOpen(false);
  };

  const handleDelete = (varName: string) => {
    const usageCount = variableUsage[varName] || 0;
    if (usageCount > 0) {
      const confirmed = window.confirm(
        `Variable "${varName}" is used in ${usageCount} node(s). ` +
        `Deleting it will remove those references. Continue?`
      );
      if (!confirmed) return;
    }

    const updatedNodes = deleteParameterFromNodes(nodes, varName);
    setNodes(updatedNodes);
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Variables ({variables.length})</CardTitle>
          <Button onClick={openCreateDialog} size="sm" variant="outline" className="h-8 px-2">
            <Plus className="w-4 h-4 mr-1" />
            Check Name
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {variables.length === 0 ? (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              No variables in use. Variables are created automatically when you reference them in nodes using the format: $variableName
            </AlertDescription>
          </Alert>
        ) : (
          <div className="max-h-96 overflow-y-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead className="w-[100px]">Used In</TableHead>
                  <TableHead className="w-[100px] text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {variables.map((varName) => (
                  <TableRow key={varName}>
                    <TableCell className="font-mono text-sm">${varName}</TableCell>
                    <TableCell className="text-sm text-gray-600">
                      {variableUsage[varName] || 0} node{variableUsage[varName] === 1 ? '' : 's'}
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-1 justify-end">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={() => openRenameDialog(varName)}
                          title="Rename variable"
                        >
                          <Edit2 className="w-3.5 h-3.5" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-destructive hover:text-destructive"
                          onClick={() => handleDelete(varName)}
                          title="Delete variable"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {dialogMode === 'create' ? 'Check Variable Name' : 'Rename Variable'}
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-4">
            {errors.length > 0 && (
              <div className="bg-destructive/10 text-destructive p-3 rounded-md text-sm">
                <ul className="list-disc list-inside space-y-1">
                  {errors.map((error, i) => (
                    <li key={i}>{error}</li>
                  ))}
                </ul>
              </div>
            )}

            {dialogMode === 'rename' && editingVariable && (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  This will rename the variable in {variableUsage[editingVariable] || 0} node(s).
                </AlertDescription>
              </Alert>
            )}

            <div className="space-y-2">
              <Label htmlFor="var-name">Variable Name</Label>
              <Input
                id="var-name"
                value={variableName}
                onChange={(e) => setVariableName(e.target.value)}
                placeholder="e.g., max_threshold, categories"
                className="font-mono"
              />
              <p className="text-xs text-muted-foreground">
                Must be a valid identifier. Use in nodes as: <code className="bg-muted px-1 py-0.5 rounded">${variableName || 'name'}</code>
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={validateAndSave}>
              {dialogMode === 'create' ? 'OK' : 'Rename'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
};