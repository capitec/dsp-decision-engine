import React, { useState, useMemo } from 'react';
import { useEditorStore } from '../editor/useEditorStore';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Database, Edit, Settings2 } from 'lucide-react';
import { UnifiedOutputTable } from '@/components/shared-tree';
import type { OutputConfiguration } from '@/components/shared-tree';
import { shiftLeafResults } from '@/utils/result-idx';
import { CustomTypeManager } from '@/components/flat-rules/CustomTypeManager';

/**
 * Output configuration tab for tree editor.
 * Compact view with button to open full editor dialog.
 */
export function OutputTab() {
  const output = useEditorStore((state) => state.output);
  const setOutput = useEditorStore((state) => state.setOutput);
  const setNodes = useEditorStore((state) => state.setNodes);
  const hasOutputErrors = useEditorStore((state) => state.hasOutputErrors);
  const [editorOpen, setEditorOpen] = useState(false);
  const [customTypesOpen, setCustomTypesOpen] = useState(false);

  const fieldCount = output.dtypes.length;
  const recordCount = output.data.length;
  const hasDefault = output.default && Object.keys(output.default).length > 0;

  // Convert TreeOutput to OutputConfiguration format for flat-rules components
  const outputConfig: OutputConfiguration = useMemo(() => ({
    dtypes: output.dtypes,
    type_defs: output.type_defs,
    output: output.data,
    default: output.default || {},
    display_fmt: output.display_fmt,
  }), [output]);

  // Handle updates from UnifiedOutputTable
  const handleConfigChange = (updates: Partial<OutputConfiguration>) => {
    // The table calls the records `output`; the document calls them `data`
    const { output: data, ...rest } = updates;
    setOutput({
      ...output,
      ...rest,
      ...(data && { data }),
      // Pass through display_fmt
      ...(updates.display_fmt !== undefined && { display_fmt: updates.display_fmt }),
    });
  };

  // Handle custom type updates
  const handleTypeDefsChange = (typeDefs: any) => {
    setOutput({
      ...output,
      type_defs: typeDefs,
    });
  };

  return (
    <>
      <Card className={hasOutputErrors ? 'ring-2 ring-red-500' : ''}>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Database className="h-4 w-4" />
            Output Configuration
            {hasOutputErrors && (
              <span className="text-red-500 text-xs">⚠ Has errors</span>
            )}
          </CardTitle>
          <CardDescription className="text-xs">
            Define output structure and data for leaf nodes
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* Summary */}
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Fields:</span>
              <span className="font-medium">{fieldCount}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Records:</span>
              <span className="font-medium">{recordCount}</span>
            </div>
            {hasDefault && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Default:</span>
                <span className="font-medium text-blue-600">Configured</span>
              </div>
            )}
          </div>

          {/* Field Preview */}
          {fieldCount > 0 && (
            <div className="border rounded p-2 max-h-32 overflow-y-auto">
              <div className="text-xs font-medium mb-1 text-muted-foreground">Fields:</div>
              <div className="space-y-1">
                {output.dtypes.slice(0, 5).map(([name, type]) => {
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

          {/* Action Buttons */}
          <div className="space-y-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setEditorOpen(true)}
              className="w-full"
            >
              <Edit className="h-4 w-4 mr-2" />
              {fieldCount === 0 ? 'Configure Output' : 'Edit Output'}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCustomTypesOpen(true)}
              className="w-full"
            >
              <Settings2 className="h-4 w-4 mr-2" />
              Manage Custom Types
              {Object.keys(output.type_defs).length > 0 && (
                <span className="ml-auto px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full text-xs">
                  {Object.keys(output.type_defs).length}
                </span>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Full Editor Dialog */}
      <Dialog open={editorOpen} onOpenChange={setEditorOpen}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Output Configuration
            </DialogTitle>
          </DialogHeader>

          <div className="flex-1 overflow-y-auto">
            <UnifiedOutputTable
              outputConfig={outputConfig}
              onConfigChange={handleConfigChange}
              onRecordRemoved={(index) => setNodes((nodes) => shiftLeafResults(nodes, index))}
              onManageCustomTypes={() => setCustomTypesOpen(true)}
            />
          </div>
        </DialogContent>
      </Dialog>

      {/* Custom Types Manager Dialog */}
      <Dialog open={customTypesOpen} onOpenChange={setCustomTypesOpen}>
        <DialogContent className="max-w-5xl h-[80vh] flex flex-col p-0">
          <DialogHeader className="px-6 pt-6 pb-4 border-b">
            <DialogTitle>Manage Custom Types</DialogTitle>
          </DialogHeader>
          <div className="flex-1 overflow-hidden">
            <CustomTypeManager
              uiTypeDefs={output.type_defs}
              outputConfig={outputConfig}
              onTypeDefsChange={handleTypeDefsChange}
              onOutputChange={(data) => handleConfigChange({ output: data })}
              onDTypesChange={(dtypes) => handleConfigChange({ dtypes })}
              onClose={() => setCustomTypesOpen(false)}
            />
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
