import React, { useState, useCallback } from 'react';
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Database, AlertCircle } from "lucide-react";
import {
  useFlatRuleProject,
  useFlatRuleSetProject,
  useFlatRuleHasInputSchemaErrors,
  useFlatRuleSectionErrors
} from './useFlatRuleStore';
import { PolarsDataType, COMMON_POLARS_TYPES } from './output-types';
import { useToast } from "@/hooks/use-toast";
import { LoadFromDatasetDialog, SchemaFieldList } from '@/components/shared-tree';

type InputSchemaField = [string, PolarsDataType];


export const InputConfigurationTab: React.FC = () => {
  const project = useFlatRuleProject();
  const setProject = useFlatRuleSetProject();
  const hasInputSchemaErrors = useFlatRuleHasInputSchemaErrors();
  const sectionErrors = useFlatRuleSectionErrors();
  const { toast } = useToast();

  // Get all input schema-related errors
  const inputSchemaErrorMessages: string[] = sectionErrors
    .filter((error: any) => error.section === 'input_schema')
    .map((error: any) => error.message);
  const [showLoadDialog, setShowLoadDialog] = useState(false);
  const [showOverwriteWarning, setShowOverwriteWarning] = useState(false);
  const [pendingSchema, setPendingSchema] = useState<InputSchemaField[] | null>(null);

  const inputSchema: InputSchemaField[] = project.project_data?.input_schema || [];

  const updateInputSchema = useCallback((newSchema: InputSchemaField[]) => {
    setProject(prev => ({
      ...prev,
      project_data: {
        ...prev.project_data,
        input_schema: newSchema
      }
    }));
  }, [setProject]);

  const handleLoadFromDataset = () => {
    if (inputSchema.length > 0) {
      setShowOverwriteWarning(true);
    } else {
      setShowLoadDialog(true);
    }
  };

  const handleConfirmOverwrite = () => {
    setShowOverwriteWarning(false);
    setShowLoadDialog(true);
  };

  const handleSchemaLoaded = (newSchema: InputSchemaField[]) => {
    if (inputSchema.length > 0) {
      setPendingSchema(newSchema);
      remapTreeFeatures(inputSchema, newSchema);
    }
    updateInputSchema(newSchema);
    setShowLoadDialog(false);
    setPendingSchema(null);
    
    toast({
      title: "Schema Loaded",
      description: `Successfully loaded ${newSchema.length} fields from dataset`,
    });
  };

  const remapTreeFeatures = (oldSchema: InputSchemaField[], newSchema: InputSchemaField[]) => {
    const oldFieldNames = oldSchema.map(([name]) => name);
    const newFieldNames = newSchema.map(([name]) => name);
    
    const fieldNameMap = new Map<string, string>();
    oldFieldNames.forEach(oldName => {
      if (newFieldNames.includes(oldName)) {
        fieldNameMap.set(oldName, oldName);
      }
    });

    if (project.project_data?.rules) {
      const updatedRules = project.project_data.rules.map(ruleRoot => ({
        ...ruleRoot,
        rule: remapNodeFeatures(ruleRoot.rule, fieldNameMap)
      }));

      setProject(prev => ({
        ...prev,
        project_data: {
          ...prev.project_data,
          rules: updatedRules
        }
      }));
    }
  };

  const remapNodeFeatures = (node: any, fieldNameMap: Map<string, string>): any => {
    if (!node) return node;

    const updatedNode = { ...node };

    if (node.feature) {
      if (fieldNameMap.has(node.feature)) {
        updatedNode.feature = fieldNameMap.get(node.feature)!;
      } else {
        updatedNode.feature = '';
      }
    }

    if (node.branches) {
      updatedNode.branches = node.branches.map((branch: any) => ({
        ...branch,
        then: remapNodeFeatures(branch.then, fieldNameMap)
      }));
    }

    if (node.otherwise) {
      updatedNode.otherwise = remapNodeFeatures(node.otherwise, fieldNameMap);
    }

    return updatedNode;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Database className="h-6 w-6" />
            Configure Inputs
          </h2>
          <p className="text-gray-600 mt-1">
            Define the input schema for your decision tree
          </p>
          {hasInputSchemaErrors && inputSchemaErrorMessages.length > 0 && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-red-900 mb-1">
                    {inputSchemaErrorMessages.length} validation {inputSchemaErrorMessages.length === 1 ? 'error' : 'errors'}:
                  </div>
                  <ul className="text-sm text-red-800 space-y-1">
                    {inputSchemaErrorMessages.map((msg, idx) => (
                      <li key={idx}>• {msg}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
        <Badge variant="outline">{inputSchema.length} fields</Badge>
      </div>

      <SchemaFieldList
        fields={inputSchema}
        onChange={updateInputSchema}
        showLoadFromDataset={true}
        onLoadFromDataset={handleLoadFromDataset}
        showSearch={true}
        emptyStateTitle="No Input Fields Defined"
        emptyStateDescription="Add fields manually or load schema from a dataset"
        addButtonLabel="Add Field"
      />

      <LoadFromDatasetDialog
        open={showLoadDialog}
        onClose={() => setShowLoadDialog(false)}
        onSchemaLoaded={handleSchemaLoaded}
      />

      <Dialog open={showOverwriteWarning} onOpenChange={setShowOverwriteWarning}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-orange-500" />
              Overwrite Current Schema?
            </DialogTitle>
            <DialogDescription>
              You already have {inputSchema.length} fields defined. Loading from a dataset will completely overwrite your current input schema.
              <br /><br />
              Any tree nodes using features that don't exist in the new schema will have their features reset to blank.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowOverwriteWarning(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleConfirmOverwrite}>
              Overwrite Schema
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};
