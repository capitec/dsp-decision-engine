import React, { useState, useCallback } from 'react';
import { useEditorStore } from '../editor/useEditorStore';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { Database, AlertCircle, Edit } from 'lucide-react';
import { useToast } from "@/hooks/use-toast";
import { LoadFromDatasetDialog, SchemaFieldList } from '@/components/shared-tree';
import type { PolarsType } from '@/types/tree-types';

type InputSchemaField = [string, PolarsType];

/**
 * Features configuration tab for tree editor.
 * Allows defining input feature schema and loading from datasets.
 */
export function FeaturesTab() {
  const inputSchema = useEditorStore((state) => state.input_schema);
  const setInputSchema = useEditorStore((state) => state.setInputSchema);
  const nodes = useEditorStore((state) => state.nodes);
  const setNodes = useEditorStore((state) => state.setNodes);
  const hasFeatureErrors = useEditorStore((state) => state.hasFeatureErrors);
  const { toast } = useToast();

  const [showLoadDialog, setShowLoadDialog] = useState(false);
  const [showOverwriteWarning, setShowOverwriteWarning] = useState(false);
  const [showEditor, setShowEditor] = useState(false);

  const fieldCount = inputSchema.length;

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
      remapNodeFeatures(inputSchema, newSchema);
    }
    setInputSchema(newSchema);
    setShowLoadDialog(false);

    toast({
      title: "Schema Loaded",
      description: `Successfully loaded ${newSchema.length} features from dataset`,
    });
  };

  const handleSchemaChange = (newSchema: InputSchemaField[]) => {
    // Remap features in nodes if names changed
    if (inputSchema.length > 0) {
      remapNodeFeatures(inputSchema, newSchema);
    }
    setInputSchema(newSchema);
  };

  const remapNodeFeatures = (oldSchema: InputSchemaField[], newSchema: InputSchemaField[]) => {
    const oldFieldNames = oldSchema.map(([name]) => name);
    const newFieldNames = newSchema.map(([name]) => name);

    const fieldNameMap = new Map<string, string>();
    oldFieldNames.forEach(oldName => {
      if (newFieldNames.includes(oldName)) {
        fieldNameMap.set(oldName, oldName);
      }
    });

    // Update all nodes that have feature references
    const updatedNodes = nodes.map(node => {
      if (node.data && 'feature' in node.data) {
        const currentFeature = (node.data as any).feature;
        if (currentFeature && !newFieldNames.includes(currentFeature)) {
          // Feature no longer exists in schema, reset to empty
          return {
            ...node,
            data: {
              ...node.data,
              feature: '',
            },
          };
        }
      }
      return node;
    });

    setNodes(updatedNodes);
  };

  return (
    <>
      <Card className={hasFeatureErrors ? 'ring-2 ring-red-500' : ''}>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Database className="h-4 w-4" />
            Features
            {hasFeatureErrors && (
              <span className="text-red-500 text-xs">⚠ Has errors</span>
            )}
          </CardTitle>
          <CardDescription className="text-xs">
            Define input features for your decision tree
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* Summary */}
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Features:</span>
              <span className="font-medium">{fieldCount}</span>
            </div>
          </div>

          {/* Field Preview */}
          {fieldCount > 0 && (
            <div className="border rounded p-2 max-h-32 overflow-y-auto">
              <div className="text-xs font-medium mb-1 text-muted-foreground">Fields:</div>
              <div className="space-y-1">
                {inputSchema.slice(0, 5).map(([name, type]) => {
                  const typeStr = typeof type === 'string' ? type : 'Custom';
                  return (
                    <div key={name} className="flex justify-between text-xs">
                      <span className="font-mono">{name}</span>
                      <span className="text-muted-foreground">{typeStr}</span>
                    </div>
                  );
                })}
                {fieldCount > 5 && (
                  <div className="text-xs text-muted-foreground italic">
                    +{fieldCount - 5} more...
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Edit Button */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowEditor(true)}
            className="w-full"
          >
            <Edit className="h-4 w-4 mr-2" />
            {fieldCount === 0 ? 'Configure Features' : 'Edit Features'}
          </Button>
        </CardContent>
      </Card>

      {/* Full Editor Dialog */}
      <Dialog open={showEditor} onOpenChange={setShowEditor}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <div className="flex items-center justify-between">
              <div>
                <DialogTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Configure Features
                </DialogTitle>
                <DialogDescription>
                  Define the input schema for your decision tree
                </DialogDescription>
              </div>
              <Badge variant="outline">{fieldCount} fields</Badge>
            </div>
          </DialogHeader>

          <div className="flex-1 overflow-y-auto">
            <SchemaFieldList
              fields={inputSchema}
              onChange={handleSchemaChange}
              showLoadFromDataset={true}
              onLoadFromDataset={handleLoadFromDataset}
              emptyStateTitle="No Features Defined"
              emptyStateDescription="Add features manually or load schema from a dataset"
              addButtonLabel="Add Feature"
            />
          </div>
        </DialogContent>
      </Dialog>

      {/* Load from Dataset Dialog */}
      <LoadFromDatasetDialog
        open={showLoadDialog}
        onClose={() => setShowLoadDialog(false)}
        onSchemaLoaded={handleSchemaLoaded}
      />

      {/* Overwrite Warning Dialog */}
      <Dialog open={showOverwriteWarning} onOpenChange={setShowOverwriteWarning}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-orange-500" />
              Overwrite Current Schema?
            </DialogTitle>
            <DialogDescription>
              You already have {inputSchema.length} features defined. Loading from a dataset will completely overwrite your current feature schema.
              <br /><br />
              Any nodes using features that don't exist in the new schema will have their features reset to blank.
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
    </>
  );
}
