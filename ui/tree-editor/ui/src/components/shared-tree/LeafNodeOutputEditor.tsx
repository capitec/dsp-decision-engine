import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { SearchableSelect } from "@/components/ui/searchable-select";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Edit2, X, AlertTriangle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { EnhancedFieldEditor } from './EnhancedFieldEditor';
import type { TreeOutput, PolarsType } from '@/types/tree-types';

export interface OutputOption {
  value: number;
  label: string;
  shortLabel: string;
}

interface LeafNodeOutputEditorProps {
  // Data bindings
  leafValue: number;
  onLeafValueChange: (value: number) => void;
  output: TreeOutput;
  outputOptions: OutputOption[];

  // Save handlers
  onSaveAsNew: (formData: Record<string, any>) => void;
  onSaveToExisting: (formData: Record<string, any>) => void;

  // Optional UI customization
  showCompactCard?: boolean; // true = full card wrapper, false = inline
  showCompactLabel?: boolean; // show "Output Selection" label in compact mode (default: true)
  showUpdateConfirmation?: boolean; // show alert dialog before saving to existing
  affectedNodesCount?: number; // number of nodes affected by update (for confirmation message)
  showTooltips?: boolean; // show tooltips on save buttons
  maxScrollHeight?: string; // tailwind class like 'max-h-96' or 'max-h-60'
  modalWidth?: 'full' | 'fixed'; // full = stretch to parent, fixed = min-w-96 max-w-lg
  compactLabel?: string; // label for compact mode card header (default: "Leaf Node")
  compactInfoText?: React.ReactNode; // info text shown in compact mode
}

/**
 * Shared Leaf Node Output Editor Component
 *
 * Features:
 * - Compact mode: dropdown + edit button (with optional card wrapper)
 * - Expanded mode: modal with field editors
 * - Save as New / Save to Existing
 * - Optional update confirmation dialog
 */
export const LeafNodeOutputEditor: React.FC<LeafNodeOutputEditorProps> = ({
  leafValue,
  onLeafValueChange,
  output,
  outputOptions,
  onSaveAsNew,
  onSaveToExisting,
  showCompactCard = true,
  showCompactLabel = true,
  showUpdateConfirmation = true,
  affectedNodesCount,
  showTooltips = true,
  maxScrollHeight = 'max-h-96',
  modalWidth = 'full',
  compactLabel = 'Leaf Node',
  compactInfoText,
}) => {
  const { toast } = useToast();

  const [isExpanded, setIsExpanded] = useState(false);
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [formInitialized, setFormInitialized] = useState(false);
  const [showUpdateConfirm, setShowUpdateConfirm] = useState(false);

  const fieldNames = output.dtypes.map(([name]) => name);

  // Helper to get field type
  const getFieldType = useCallback((fieldName: string): PolarsType => {
    const entry = output.dtypes.find(([name]) => name === fieldName);
    return (entry ? entry[1] : 'String') as PolarsType;
  }, [output.dtypes]);

  // Initialize form data when modal opens
  useEffect(() => {
    if (isExpanded && !formInitialized) {
      const currentOutputIndex = leafValue ?? -1;

      if (currentOutputIndex >= 0 && currentOutputIndex < output.data.length) {
        setFormData({ ...output.data[currentOutputIndex] });
      } else if (currentOutputIndex === -1) {
        setFormData({ ...(output.default || {}) });
      } else {
        // Initialize with empty data
        const emptyData: Record<string, any> = {};
        fieldNames.forEach(fieldName => {
          const fieldType = getFieldType(fieldName);
          if (typeof fieldType === 'string' && (fieldType.includes('Int') || fieldType.includes('Float'))) {
            emptyData[fieldName] = null;
          } else if (fieldType === 'Boolean') {
            emptyData[fieldName] = false;
          } else {
            emptyData[fieldName] = '';
          }
        });
        setFormData(emptyData);
      }
      setFormInitialized(true);
    }

    // Reset initialization flag when modal closes
    if (!isExpanded && formInitialized) {
      setFormInitialized(false);
    }
  }, [isExpanded, formInitialized, leafValue, output.data, output.default, fieldNames, getFieldType]);

  // Handle output selection change
  const handleOutputChange = useCallback((value: string) => {
    const index = parseInt(value);
    onLeafValueChange(index);
  }, [onLeafValueChange]);

  // Convert output options to SearchableSelect format
  const searchableOptions = outputOptions.map(option => ({
    value: String(option.value),
    label: (
      <div className="flex items-center gap-2 min-w-0">
        <span className="font-mono text-xs bg-gray-100 px-1 rounded shrink-0">
          {option.shortLabel}
        </span>
        <span className="truncate">{option.label}</span>
      </div>
    ),
    searchLabel: `${option.shortLabel} ${option.label}`, // Plain text for search
  }));

  // Handle form field change
  const handleFieldChange = useCallback((fieldName: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [fieldName]: value
    }));
  }, []);

  // Handle save as new output
  const handleSaveAsNew = useCallback(() => {
    onSaveAsNew(formData);
    setIsExpanded(false);
  }, [formData, onSaveAsNew]);

  // Handle save to existing (with optional confirmation)
  const handleSaveToExisting = useCallback(() => {
    if (showUpdateConfirmation && affectedNodesCount !== undefined && affectedNodesCount > 0) {
      setShowUpdateConfirm(true);
    } else {
      confirmUpdate();
    }
  }, [showUpdateConfirmation, affectedNodesCount]);

  // Actually perform the update (after confirmation if enabled)
  const confirmUpdate = useCallback(() => {
    onSaveToExisting(formData);
    setShowUpdateConfirm(false);
    setIsExpanded(false);
  }, [formData, onSaveToExisting]);

  const canExpand = fieldNames.length > 0;
  const selectedOption = outputOptions.find(opt => opt.value === leafValue);

  // Compact mode - dropdown and edit button
  const compactContent = (
    <div className={showCompactLabel ? "space-y-2" : ""}>
      {showCompactLabel && <Label>Output Selection</Label>}
      <div className="flex items-center gap-2">
        <SearchableSelect
          options={searchableOptions}
          value={String(leafValue ?? -1)}
          onValueChange={handleOutputChange}
          placeholder="Select output..."
          searchPlaceholder="Search outputs..."
          emptyMessage="No outputs found."
          triggerClassName={showCompactCard ? "flex-1" : "w-40"}
        />

        {canExpand && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(true)}
            className={showCompactCard ? "p-2 h-9" : "p-1 h-7 w-7"}
            title="Edit output values"
          >
            <Edit2 className={showCompactCard ? "h-4 w-4" : "h-3 w-3"} />
          </Button>
        )}
      </div>
    </div>
  );

  // Compact mode - just inline controls
  if (!isExpanded && !showCompactCard) {
    return <div className="flex items-center gap-2">{compactContent}</div>;
  }

  // Compact mode - full card wrapper
  if (!isExpanded && showCompactCard) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">{compactLabel}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {compactContent}

          {compactInfoText && (
            <div className="pt-2 border-t">
              <div className="text-xs text-gray-500">
                {compactInfoText}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  }

  // Expanded mode - full editor in a modal-like overlay
  const modalWidthClass = modalWidth === 'full'
    ? 'absolute top-0 left-0 right-0'
    : 'absolute top-0 left-0 min-w-96 max-w-lg';

  return (
    <div className="relative">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 z-40"
        onClick={() => setIsExpanded(false)}
      />

      {/* Modal Card */}
      <div className={`${modalWidthClass} z-50 bg-white border border-gray-300 rounded-lg shadow-lg`}>
        <Card className="border-0 shadow-none">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span>Edit Output:</span>
                <span className="font-mono text-xs bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded">
                  {selectedOption?.shortLabel || '0'}
                </span>
                <span className="text-blue-600 font-normal truncate">
                  {selectedOption?.label || 'Unknown'}
                </span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsExpanded(false)}
                className="p-1 h-6 w-6"
              >
                <X className="h-3 w-3" />
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {canExpand && (
              <>
                <div className={`space-y-3 ${maxScrollHeight} overflow-y-auto`}>
                  {fieldNames.map((fieldName) => (
                    <div key={fieldName} className="space-y-1">
                      <Label htmlFor={`field-${fieldName.replace(/[^a-zA-Z0-9]/g, '-')}`} className="text-xs">
                        {fieldName}
                      </Label>
                      <EnhancedFieldEditor
                        fieldName={fieldName}
                        fieldType={getFieldType(fieldName)}
                        value={formData[fieldName]}
                        outputConfig={{ dtypes: output.dtypes, type_defs: output.type_defs, output: output.data, default: output.default || {} }}
                        onChange={(value) => handleFieldChange(fieldName, value)}
                      />
                    </div>
                  ))}
                </div>

                <div className="space-y-2 pt-2 border-t">
                  {showTooltips ? (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={handleSaveAsNew}
                            className="w-full text-xs"
                          >
                            Save as New
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Create a new output record and switch this node to use it</p>
                        </TooltipContent>
                      </Tooltip>

                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            size="sm"
                            onClick={handleSaveToExisting}
                            className="w-full text-xs"
                          >
                            Save to Existing
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>
                            Update this output record
                            {affectedNodesCount !== undefined && ` (affects ${affectedNodesCount} leaf nodes)`}
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  ) : (
                    <>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={handleSaveAsNew}
                        className="w-full text-xs"
                      >
                        Save as New
                      </Button>
                      <Button
                        size="sm"
                        onClick={handleSaveToExisting}
                        className="w-full text-xs"
                      >
                        Save to Existing
                      </Button>
                    </>
                  )}
                </div>
              </>
            )}

            {!canExpand && (
              <div className="text-xs text-gray-500">
                <p>No output schema configured. Configure outputs in the Output tab to enable editing.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Update Confirmation Dialog */}
      {showUpdateConfirmation && (
        <AlertDialog open={showUpdateConfirm} onOpenChange={setShowUpdateConfirm}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-orange-500" />
                Update Existing Output?
              </AlertDialogTitle>
              <AlertDialogDescription>
                This will update output record{' '}
                <span className="font-mono font-semibold">
                  {leafValue === -1 ? 'default' : leafValue}
                </span>
                {' '}for all leaf nodes that reference it.
                {affectedNodesCount !== undefined && (
                  <>
                    <br /><br />
                    <span className="font-semibold">{affectedNodesCount} leaf node(s)</span> will be affected by this change.
                  </>
                )}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction onClick={confirmUpdate}>
                Update
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  );
};
