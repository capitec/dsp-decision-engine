import React, { useState, useMemo, useEffect } from 'react';
import { paramNameError } from "@/utils/param-names";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Plus, Trash2, AlertCircle,
  Settings, Info, Search, X
} from "lucide-react";
import { EnhancedFieldEditor } from './EnhancedFieldEditor';
import { UnifiedTypeSelector } from './UnifiedTypeSelector';
import { PolarsDataType, COMMON_POLARS_TYPES } from '@/types/datasource-types';

export interface Parameter {
  id: string;
  name: string;
  type: PolarsDataType;
  value: any;
}

export interface ParameterManagementDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  parameters: Parameter[];
  onSave: (parameters: Parameter[]) => void; // Changed to batch save
  isParameterInUse: (name: string) => boolean;
  title?: string;
  description?: string;
}

// Helper to get default value for a parameter type
const getDefaultValueForType = (type: PolarsDataType): any => {
  const typeStr = typeof type === 'string' ? type : (type as any).type;
  if (typeStr === 'List') return [];
  if (typeStr.includes('Float')) return 0.0;
  if (typeStr.includes('Int')) return 0;
  if (typeStr === 'Boolean') return false;
  if (typeStr === 'Date') return new Date().toISOString().split('T')[0];
  return '';
};

// Helper to get type display color
const getTypeColor = (type: PolarsDataType): string => {
  const typeStr = typeof type === 'string' ? type : (type as any).type;
  if (typeStr === 'List') return 'bg-indigo-100 text-indigo-800';
  if (typeStr.includes('Int') || typeStr.includes('UInt')) return 'bg-blue-100 text-blue-800';
  if (typeStr.includes('Float')) return 'bg-cyan-100 text-cyan-800';
  if (typeStr === 'Boolean') return 'bg-purple-100 text-purple-800';
  if (typeStr === 'String') return 'bg-green-100 text-green-800';
  if (typeStr === 'Date' || typeStr === 'Datetime' || typeStr === 'Time') return 'bg-orange-100 text-orange-800';
  return 'bg-gray-100 text-gray-800';
};

/**
 * Generic parameter management dialog that can be used by both tree editor and flat-rules.
 *
 * For tree editor: manages variables as parameters
 * For flat-rules: manages project parameters
 *
 * Uses draft state with dialog-level Save/Cancel buttons.
 * Prevents renaming/deleting parameters that are in use.
 */
export const ParameterManagementDialog: React.FC<ParameterManagementDialogProps> = ({
  open,
  onOpenChange,
  parameters,
  onSave,
  isParameterInUse,
  title = "Manage Parameters",
  description = "Create and manage reusable parameters for your decision tree"
}) => {
  // Draft state - changes are not saved until user clicks Save
  const [draftParameters, setDraftParameters] = useState<Parameter[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [showNewRow, setShowNewRow] = useState(false);
  const [newParamName, setNewParamName] = useState('');
  const [newParamType, setNewParamType] = useState<PolarsDataType>('String');
  const [newParamValue, setNewParamValue] = useState<any>('');
  const [errors, setErrors] = useState<string[]>([]);

  // Initialize draft state when dialog opens
  useEffect(() => {
    if (open) {
      setDraftParameters(JSON.parse(JSON.stringify(parameters))); // Deep clone
      setSearchQuery('');
      setShowNewRow(false);
      setNewParamName('');
      setNewParamType('String');
      setNewParamValue('');
      setErrors([]);
    }
  }, [open, parameters]);

  // Filter draft parameters based on search
  const filteredParameters = useMemo(() => {
    if (!searchQuery.trim()) return draftParameters;

    const query = searchQuery.toLowerCase();
    return draftParameters.filter(param =>
      param.name.toLowerCase().includes(query) ||
      (typeof param.type === 'string' && param.type.toLowerCase().includes(query))
    );
  }, [draftParameters, searchQuery]);

  const handleAddParameter = () => {
    const newErrors: string[] = [];

    // Validate name
    const nameError = paramNameError(newParamName);
    if (nameError) {
      newErrors.push(nameError);
    }

    // Check for duplicate names in draft
    if (draftParameters.some(p => p.name === newParamName.trim())) {
      newErrors.push('Parameter name must be unique');
    }

    if (newErrors.length > 0) {
      setErrors(newErrors);
      return;
    }

    // Create parameter with appropriate value based on type
    const defaultValue = newParamValue || getDefaultValueForType(newParamType);

    // Convert simple "List" string to proper PolarsListType structure
    // Default to List<Int32> for categorical compatibility
    const typeStr = typeof newParamType === 'string' ? newParamType : (newParamType as any).type;
    const finalType = typeStr === 'List'
      ? { type: 'List' as const, inner: 'Int32' as const }
      : newParamType;

    // Add to draft
    const newParam: Parameter = {
      id: `param_${Date.now()}`,
      name: newParamName.trim(),
      type: finalType,
      value: defaultValue
    };
    setDraftParameters([...draftParameters, newParam]);

    // Reset form
    setNewParamName('');
    setNewParamType('String');
    setNewParamValue('');
    setShowNewRow(false);
    setErrors([]);
  };

  const handleUpdateParameter = (paramId: string, updates: Partial<Parameter>) => {
    setDraftParameters(draftParameters.map(p =>
      p.id === paramId ? { ...p, ...updates } : p
    ));
  };

  const handleDeleteParameter = (paramId: string, paramName: string) => {
    if (confirm(`Are you sure you want to delete parameter "${paramName}"?`)) {
      setDraftParameters(draftParameters.filter(p => p.id !== paramId));
    }
  };

  const handleSave = () => {
    // Validate all parameters before saving
    const errors: string[] = [];
    const nameSet = new Set<string>();

    const trimmed = draftParameters.map(p => ({ ...p, name: p.name.trim() }));
    for (const param of trimmed) {
      const nameError = paramNameError(param.name);
      if (nameError) {
        errors.push(nameError);
        break;
      }
      if (nameSet.has(param.name)) {
        errors.push(`Duplicate parameter name: ${param.name}`);
        break;
      }
      nameSet.add(param.name);
    }

    if (errors.length > 0) {
      setErrors(errors);
      return;
    }

    onSave(trimmed);
    onOpenChange(false);
  };

  const handleCancel = () => {
    onOpenChange(false);
  };

  // Filter available types (exclude complex types for parameters)
  const availableTypes = COMMON_POLARS_TYPES.filter(t =>
    t !== 'Categorical' && t !== 'Decimal' && t.trim() !== ''
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[80vh] flex flex-col p-0">
        <DialogHeader className="px-6 pt-6 pb-4 border-b">
          <DialogTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            {title}
          </DialogTitle>
          <p className="text-sm text-muted-foreground mt-1">
            {description}
          </p>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {/* Info Card */}
          <Card className="border-blue-200 bg-blue-50/30">
            <CardContent className="pt-4">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <div className="text-sm text-blue-800">
                  <p className="font-medium mb-1">How Parameters Work</p>
                  <div>
                    Parameters allow you to define reusable values that can be referenced in your nodes.
                    Use the <Badge variant="outline" className="mx-1 text-xs">#</Badge> button to
                    reference parameters instead of hardcoded values.
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Search Bar */}
          {parameters.length > 0 && (
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search parameters by name or type..."
                className="pl-9 pr-9"
              />
              {searchQuery && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSearchQuery('')}
                  className="absolute right-1 top-1/2 -translate-y-1/2 h-6 w-6 p-0"
                >
                  <X className="h-3 w-3" />
                </Button>
              )}
            </div>
          )}

          {/* Validation Errors */}
          {errors.length > 0 && (
            <Card className="border-red-200 bg-red-50">
              <CardContent className="pt-4">
                <div className="flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                  <div>
                    {errors.map((error, index) => (
                      <div key={index} className="text-sm text-red-700">{error}</div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Parameters Table */}
          {draftParameters.length > 0 || showNewRow ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Parameters ({filteredParameters.length})
                  {searchQuery && filteredParameters.length < draftParameters.length && (
                    <span className="text-sm font-normal text-gray-500">
                      (filtered from {draftParameters.length})
                    </span>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="border rounded-lg overflow-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50 border-b">
                      <tr>
                        <th className="text-left p-3 font-medium text-sm w-1/4">Name</th>
                        <th className="text-left p-3 font-medium text-sm w-1/6">Type</th>
                        <th className="text-left p-3 font-medium text-sm w-1/2">Value</th>
                        <th className="text-left p-3 font-medium text-sm w-1/12">In Use</th>
                        <th className="text-center p-3 font-medium text-sm w-20">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                  {/* New Parameter Row */}
                  {showNewRow && (
                    <tr className="border-b bg-blue-50/30">
                      <td className="p-2">
                        <Input
                          value={newParamName}
                          onChange={(e) => setNewParamName(e.target.value)}
                          placeholder="parameter_name"
                          className="w-full"
                          autoFocus
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              handleAddParameter();
                            } else if (e.key === 'Escape') {
                              setShowNewRow(false);
                              setNewParamName('');
                              setNewParamType('String');
                              setNewParamValue('');
                              setErrors([]);
                            }
                          }}
                        />
                      </td>
                      <td className="p-2">
                        <UnifiedTypeSelector
                          value={newParamType}
                          onValueChange={(type) => {
                            setNewParamType(type);
                            setNewParamValue(getDefaultValueForType(type));
                          }}
                          showSearch={true}
                          enableComplexTypes={true}
                          className="w-full"
                        />
                      </td>
                      <td className="p-2">
                        <EnhancedFieldEditor
                          fieldType={newParamType}
                          value={newParamValue}
                          onChange={setNewParamValue}
                          allowNaN={true}
                          allowInfinity={true}
                          allowEmpty={false}
                          className="w-full"
                        />
                      </td>
                      <td className="p-2"></td>
                      <td className="p-2 text-center">
                        <div className="flex items-center justify-center gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={handleAddParameter}
                            className="text-green-600 hover:text-green-700 p-1 h-6 text-xs"
                          >
                            Add
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => {
                              setShowNewRow(false);
                              setNewParamName('');
                              setNewParamType('String');
                              setNewParamValue('');
                              setErrors([]);
                            }}
                            className="text-gray-600 hover:text-gray-700 p-1 h-6 text-xs"
                          >
                            Cancel
                          </Button>
                        </div>
                      </td>
                    </tr>
                  )}

                  {/* Existing Parameters */}
                  {filteredParameters.length === 0 && !showNewRow ? (
                    <tr>
                      <td colSpan={5} className="p-8 text-center text-gray-500">
                        {searchQuery ? 'No parameters match your search' : 'No parameters yet'}
                      </td>
                    </tr>
                  ) : (
                    filteredParameters.map((parameter) => {
                      const inUse = isParameterInUse(parameter.name);
                      return (
                        <tr key={parameter.id} className="border-b hover:bg-gray-50">
                          <td className="p-2">
                            <Input
                              value={parameter.name}
                              onChange={(e) => handleUpdateParameter(parameter.id, { name: e.target.value })}
                              disabled={inUse}
                              title={inUse ? "Cannot rename parameter that is in use" : "Parameter name"}
                              className="w-full font-medium disabled:opacity-60 disabled:cursor-not-allowed"
                            />
                            {paramNameError(parameter.name) && (
                              <p className="text-xs text-red-600 mt-1">{paramNameError(parameter.name)}</p>
                            )}
                          </td>
                          <td className="p-2">
                            <UnifiedTypeSelector
                              value={parameter.type}
                              onValueChange={(type) => handleUpdateParameter(parameter.id, { type })}
                              disabled={inUse}
                              showSearch={true}
                              enableComplexTypes={true}
                              className="w-full"
                              triggerClassName={inUse ? "opacity-60 cursor-not-allowed" : ""}
                            />
                          </td>
                          <td className="p-2">
                            <EnhancedFieldEditor
                              fieldType={parameter.type}
                              value={parameter.value}
                              onChange={(value) => handleUpdateParameter(parameter.id, { value })}
                              allowNaN={true}
                              allowInfinity={true}
                              allowEmpty={false}
                              className="w-full"
                            />
                          </td>
                          <td className="p-2 text-center">
                            {inUse && (
                              <Badge variant="outline" className="text-xs text-green-600 border-green-300">
                                Yes
                              </Badge>
                            )}
                          </td>
                          <td className="p-2 text-center">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleDeleteParameter(parameter.id, parameter.name)}
                              disabled={inUse}
                              title={inUse ? "Cannot delete parameter that is in use" : "Delete parameter"}
                              className="text-red-600 hover:text-red-700 disabled:text-gray-400 p-1 h-8 w-8"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="border-dashed border-gray-300">
              <CardContent className="pt-8 pb-8 text-center">
                <Settings className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Parameters Yet</h3>
                <p className="text-gray-600 mb-4">
                  Create your first parameter to start using reusable values.
                </p>
                <Button onClick={() => setShowNewRow(true)}>
                  <Plus className="w-4 h-4 mr-2" />
                  Add Your First Parameter
                </Button>
              </CardContent>
            </Card>
          )}
        </div>

        <DialogFooter className="px-6 py-4 border-t bg-gray-50 flex justify-between items-center">
          <Button
            onClick={() => setShowNewRow(true)}
            disabled={showNewRow}
            variant="outline"
            size="sm"
          >
            <Plus className="w-4 h-4 mr-2" />
            Add Parameter
          </Button>
          <div className="flex gap-2">
            <Button onClick={handleCancel} variant="outline">
              Cancel
            </Button>
            <Button onClick={handleSave}>
              Save Changes
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
