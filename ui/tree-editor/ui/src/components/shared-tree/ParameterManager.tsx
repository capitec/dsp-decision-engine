import { useState } from 'react';
import { Plus, Trash2, Edit2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import type { ParameterMap, UITypeDefs } from '@/types/tree-types';
import { UnifiedTypeSelector } from './UnifiedTypeSelector';
import { EnhancedFieldEditor } from './EnhancedFieldEditor';

interface ParameterManagerProps {
  parameters: ParameterMap;
  onChange: (parameters: ParameterMap) => void;
  typeDefs: UITypeDefs; // From output.type_defs
  className?: string;
}

/**
 * Shared parameter management component.
 *
 * Features:
 * - Add/edit/delete parameters
 * - Type selection using UnifiedTypeSelector from flat-rules
 * - Default value editing with EnhancedFieldEditor from flat-rules
 * - Parameter name validation
 */
export function ParameterManager({
  parameters,
  onChange,
  typeDefs,
  className,
}: ParameterManagerProps) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingParam, setEditingParam] = useState<string | null>(null);
  const [paramName, setParamName] = useState('');
  const [paramType, setParamType] = useState<any>('String');
  const [paramDefaultValue, setParamDefaultValue] = useState<any>(null);
  const [errors, setErrors] = useState<string[]>([]);

  const openCreateDialog = () => {
    setEditingParam(null);
    setParamName('');
    setParamType('String');
    setParamDefaultValue(null);
    setErrors([]);
    setDialogOpen(true);
  };

  const openEditDialog = (name: string) => {
    const param = parameters[name];
    if (!param) return;

    setEditingParam(name);
    setParamName(name);
    // Convert PolarsType to string for UI (primitives only for parameters)
    setParamType(typeof param.type === 'string' ? param.type : 'String');
    setParamDefaultValue(param.default_value);
    setErrors([]);
    setDialogOpen(true);
  };

  const validateAndSave = () => {
    const newErrors: string[] = [];

    // Validate name
    const trimmedName = paramName.trim();
    if (!trimmedName) {
      newErrors.push('Parameter name is required');
    }
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(trimmedName)) {
      newErrors.push('Parameter name must be a valid identifier (letters, numbers, underscores, cannot start with number)');
    }

    // Check for duplicate name (only in create mode or if name changed)
    if (editingParam !== trimmedName && parameters[trimmedName]) {
      newErrors.push(`Parameter "${trimmedName}" already exists`);
    }

    if (newErrors.length > 0) {
      setErrors(newErrors);
      return;
    }

    // Create or update parameter
    const newParameters = { ...parameters };

    // If renaming, remove old key
    if (editingParam && editingParam !== trimmedName) {
      delete newParameters[editingParam];
    }

    // UnifiedTypeSelector only allows primitive types for parameters (no Decimal, List, Categorical)
    // So paramType is always a primitive string, which is a valid PolarsType
    newParameters[trimmedName] = {
      type: paramType as any, // Primitive string types are valid PolarsType
      default_value: paramDefaultValue,
    };

    onChange(newParameters);
    setDialogOpen(false);
  };

  const deleteParameter = (name: string) => {
    if (window.confirm(`Are you sure you want to delete parameter "${name}"?`)) {
      const newParameters = { ...parameters };
      delete newParameters[name];
      onChange(newParameters);
    }
  };

  const paramEntries = Object.entries(parameters);

  return (
    <div className={className}>
      <div className="space-y-4">
        <div className="flex items-center justify-end">
          <Button onClick={openCreateDialog} size="sm">
            <Plus className="w-4 h-4 mr-2" />
            Add Parameter
          </Button>
        </div>

        {paramEntries.length === 0 ? (
          <div className="text-center text-muted-foreground py-8 border-2 border-dashed rounded-md">
            No parameters defined. Parameters allow runtime configuration of your tree.
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Default Value</TableHead>
                <TableHead className="w-[100px]">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {paramEntries.map(([name, param]) => (
                <TableRow key={name}>
                  <TableCell className="font-mono">{name}</TableCell>
                  <TableCell>{typeof param.type === 'string' ? param.type : 'String'}</TableCell>
                  <TableCell>
                    <code className="text-sm">{JSON.stringify(param.default_value)}</code>
                  </TableCell>
                  <TableCell>
                    <div className="flex gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => openEditDialog(name)}
                      >
                        <Edit2 className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => deleteParameter(name)}
                        className="text-destructive"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {editingParam ? 'Edit Parameter' : 'Create Parameter'}
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-4">
            {errors.length > 0 && (
              <div className="bg-destructive/10 text-destructive p-3 rounded-md">
                <ul className="list-disc list-inside">
                  {errors.map((error, i) => (
                    <li key={i}>{error}</li>
                  ))}
                </ul>
              </div>
            )}

            <div>
              <Label htmlFor="param-name">Parameter Name</Label>
              <Input
                id="param-name"
                value={paramName}
                onChange={(e) => setParamName(e.target.value)}
                placeholder="e.g., max_age, category_threshold"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Must be a valid identifier (letters, numbers, underscores)
              </p>
            </div>

            <div>
              <Label htmlFor="param-type">Type</Label>
              <UnifiedTypeSelector
                value={paramType}
                onValueChange={setParamType}
                excludeTypes={['Categorical', 'Decimal', 'List']}
                className="w-full"
              />
            </div>

            <div>
              <Label htmlFor="param-default">Default Value</Label>
              <EnhancedFieldEditor
                fieldType={paramType as any}
                value={paramDefaultValue}
                onChange={setParamDefaultValue}
                allowNaN={true}
                allowInfinity={true}
                allowEmpty={false}
                className="w-full"
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={validateAndSave}>
              {editingParam ? 'Save' : 'Create'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
