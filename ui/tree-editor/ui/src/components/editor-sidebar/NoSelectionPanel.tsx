import React, { useCallback, useState } from 'react';
import { useReactFlow } from '@xyflow/react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { ArrowUpDown, Settings } from "lucide-react";
import { useEditorContext } from '../editor/EditorContext';
import { useEditorStore } from '../editor/useEditorStore';
import { SubtreeManager } from '../editor/SubtreeManager';
import { OutputTab } from './OutputTab';
import { FeaturesTab } from './FeaturesTab';
import { ParameterManagementDialog } from '../shared-tree/ParameterManagementDialog';
import { useParameterManagement } from '../editor/useParameterManagement';

export const NoSelectionPanel: React.FC = () => {
  const { handleRearrange } = useEditorContext();
  const autoRearrange = useEditorStore((state) => state.autoRearrange);
  const setAutoRearrange = useEditorStore((state) => state.setAutoRearrange);
  const addNode = useEditorStore((state) => state.addNode);
  const hasParameterErrors = useEditorStore((state) => state.hasParameterErrors);

  const [parametersDialogOpen, setParametersDialogOpen] = useState(false);

  const { getViewport } = useReactFlow();

  // Parameter management
  const {
    parametersForDialog,
    isParameterInUse,
    saveParameters,
    parameterCount
  } = useParameterManagement();

  // Add new node to center of viewport
  const handleAddNode = useCallback(() => {
    const viewport = getViewport();

    // Calculate center position of current viewport
    const centerX = (-viewport.x + (window.innerWidth - 320) / 2) / viewport.zoom; // 320 is sidebar width
    const centerY = (-viewport.y + window.innerHeight / 2) / viewport.zoom;

    const newNode = {
      id: `node_${Date.now()}`,
      type: 'default' as const,
      position: { x: centerX, y: centerY },
      data: {
        type: 'leaf' as const,
        result_idx: -1,
      }
    };

    addNode(newNode);
  }, [addNode, getViewport]);

  return (
    <>
      {/* General Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">General</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <Button
            onClick={handleAddNode}
            className="w-full"
            size="sm"
          >
            Add Node
          </Button>

          <Button
            onClick={() => setParametersDialogOpen(true)}
            variant="outline"
            className={`w-full ${hasParameterErrors ? 'ring-2 ring-red-500' : ''}`}
            size="sm"
          >
            <Settings className="w-4 h-4 mr-2" />
            Manage Parameters
            {hasParameterErrors && (
              <span className="ml-2 text-red-500 text-xs">⚠</span>
            )}
            {parameterCount > 0 && (
              <span className="ml-2 px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full text-xs">
                {parameterCount}
              </span>
            )}
          </Button>

          <Button
            onClick={handleRearrange}
            variant="outline"
            className="w-full"
            size="sm"
          >
            <ArrowUpDown className="w-4 h-4 mr-2" />
            Rearrange Tree
          </Button>

          <div className="flex items-center space-x-2">
            <Checkbox
              id="auto-arrange"
              checked={autoRearrange}
              onCheckedChange={(checked) => setAutoRearrange(checked === true)}
            />
            <Label htmlFor="auto-arrange" className="text-sm">
              Auto-arrange tree
            </Label>
          </div>
        </CardContent>
      </Card>

      {/* Features & Output */}
      <FeaturesTab />
      <OutputTab />

      {/* Subtree Manager */}
      <SubtreeManager />

      {/* Parameter Management Dialog */}
      <ParameterManagementDialog
        open={parametersDialogOpen}
        onOpenChange={setParametersDialogOpen}
        parameters={parametersForDialog}
        onSave={saveParameters}
        isParameterInUse={isParameterInUse}
        title="Manage Variables"
        description="Variables are defined inline in nodes. Create variables by using $variableName in node editors."
      />
    </>
  );
};