import React, { useState, useCallback } from 'react';
import { shiftLeafResults } from '@/utils/result-idx';
import { Database, Settings2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import {
  useFlatRuleProject,
  useFlatRuleSetProject,
  useFlatRuleSetTrees,
  useFlatRuleHasOutputErrors,
  useFlatRuleSectionErrors
} from './useFlatRuleStore';
import { UnifiedOutputTable } from '@/components/shared-tree';
import type { OutputConfiguration } from '@/components/shared-tree';
import { CustomTypeManager } from './CustomTypeManager';
import {
  UITypeDefs
} from './output-types';
import type { FlatRuleProjectDetail, PEDPrioritizedTree } from './types';

export const OutputConfigurationTab: React.FC = () => {
  const project = useFlatRuleProject();
  const setProject = useFlatRuleSetProject();
  const setTrees = useFlatRuleSetTrees();
  const [customTypesOpen, setCustomTypesOpen] = useState(false);
  const hasOutputErrors = useFlatRuleHasOutputErrors();
  const sectionErrors = useFlatRuleSectionErrors();

  // Get all output-related errors
  const outputErrorMessages: string[] = sectionErrors
    .filter(error => error.section === 'output')
    .map(error => error.message);

  // Get output configuration from project data (nested under output field)
  const outputConfig: OutputConfiguration = {
    dtypes: project.project_data?.output?.dtypes || [],
    type_defs: (project.project_data?.output?.type_defs || {}) as UITypeDefs,
    output: project.project_data?.output?.data || [],
    default: project.project_data?.output?.default || {},
    display_fmt: project.project_data?.output?.display_fmt
  };

  // Update output configuration in project data (nested under output field)
  const updateOutputConfig = useCallback((updates: Partial<OutputConfiguration>) => {
    setProject((prev): FlatRuleProjectDetail => ({
      ...prev,
      project_data: {
        ...prev.project_data,
        output: {
          ...(prev.project_data?.output || {}),
          // Map output config keys to nested structure
          ...(updates.dtypes && { dtypes: updates.dtypes }),
          ...(updates.type_defs && { type_defs: updates.type_defs }),
          ...(updates.output && { data: updates.output }),
          ...(updates.default && { default: updates.default }),
          ...(updates.display_fmt !== undefined && { display_fmt: updates.display_fmt }),
        }
      } as PEDPrioritizedTree
    }));
  }, [setProject]);

  const handleTypeDefsChange = useCallback((typeDefs: any) => {
    updateOutputConfig({ type_defs: typeDefs });
  }, [updateOutputConfig]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Database className="h-6 w-6" />
            Output Configuration
          </h2>
          <p className="text-gray-600 mt-1">
            Configure the structure, types, and data for your decision tree outputs
          </p>
          {hasOutputErrors && outputErrorMessages.length > 0 && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-red-900 mb-1">
                    {outputErrorMessages.length} validation {outputErrorMessages.length === 1 ? 'error' : 'errors'}:
                  </div>
                  <ul className="text-sm text-red-800 space-y-1">
                    {outputErrorMessages.map((msg, idx) => (
                      <li key={idx}>• {msg}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
        <Button variant="outline" size="sm" onClick={() => setCustomTypesOpen(true)}>
          <Settings2 className="h-4 w-4 mr-2" />
          Manage Custom Types
          {Object.keys(outputConfig.type_defs).length > 0 && (
            <span className="ml-2 px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full text-xs">
              {Object.keys(outputConfig.type_defs).length}
            </span>
          )}
        </Button>
      </div>

      {/* Unified Data & Schema */}
      <UnifiedOutputTable
        outputConfig={outputConfig}
        onConfigChange={updateOutputConfig}
        onRecordRemoved={(index) =>
          setTrees((trees) => trees.map((t) => ({ ...t, rootNode: shiftLeafResults(t.rootNode, index) })))
        }
        onManageCustomTypes={() => setCustomTypesOpen(true)}
      />

      {/* Custom Types Dialog */}
      <Dialog open={customTypesOpen} onOpenChange={setCustomTypesOpen}>
        <DialogContent className="max-w-5xl h-[80vh] flex flex-col p-0">
          <DialogHeader className="px-6 pt-6 pb-4 border-b">
            <DialogTitle className="flex items-center gap-2">
              <Settings2 className="h-5 w-5" />
              Manage Custom Types
            </DialogTitle>
          </DialogHeader>
          <div className="flex-1 overflow-hidden">
            <CustomTypeManager
              uiTypeDefs={outputConfig.type_defs}
              outputConfig={outputConfig}
              onTypeDefsChange={handleTypeDefsChange}
              onOutputChange={(output) => updateOutputConfig({ output })}
              onDTypesChange={(dtypes) => updateOutputConfig({ dtypes })}
              onClose={() => setCustomTypesOpen(false)}
            />
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};