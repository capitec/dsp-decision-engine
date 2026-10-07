import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { CheckCircle, AlertCircle, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export interface NodeValidationResult {
  node_id: string;
  node_type?: string;
  errors: string[];
  warnings: string[];
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  format_version?: number;
  node_count?: number;
  feature_count?: number;
  subtree_count?: number;
  node_results?: NodeValidationResult[];
}

/**
 * Parse and improve validation error messages for better readability
 */
const parseErrorMessage = (error: string): string => {
  // Handle Pydantic validation errors with complex structure
  if (error.includes("'type': 'value_error'")) {
    try {
      // Extract the main message from the error structure
      const msgMatch = error.match(/'msg': '([^']+)'/);
      if (msgMatch) {
        let message = msgMatch[1];
        
        // Clean up common patterns
        message = message.replace(/Value error, /, '');
        message = message.replace(/Tree output validation failed:\\n/, '');
        message = message.replace(/\\n/g, '\n');
        
        // Extract node ID and make it more readable
        const nodeMatch = message.match(/node: (node_\d+)/);
        if (nodeMatch) {
          const nodeId = nodeMatch[1];
          message = message.replace(nodeId, `Node "${nodeId}"`);
        }
        
        return message;
      }
    } catch (e) {
      // If parsing fails, fall back to original
    }
  }
  
  // Handle other common error patterns
  if (error.includes('Output data of node:') && error.includes('is missing')) {
    const nodeMatch = error.match(/node: (node_\d+)/);
    const nodeId = nodeMatch ? nodeMatch[1] : 'unknown';
    return `Node "${nodeId}" is missing output data and no default value is set. Please configure the leaf value or set a default output.`;
  }
  
  // Clean up general formatting
  return error
    .replace(/\\n/g, '\n')
    .replace(/Tree output validation failed:\n/, '')
    .replace(/Value error, /, '');
};

/**
 * Parse and improve validation results for better readability
 */
const parseValidationResults = (results: ValidationResult): ValidationResult => {
  return {
    ...results,
    errors: results.errors.map(parseErrorMessage),
    warnings: results.warnings.map(parseErrorMessage),
    node_results: results.node_results?.map(nodeResult => ({
      ...nodeResult,
      errors: nodeResult.errors.map(parseErrorMessage),
      warnings: nodeResult.warnings.map(parseErrorMessage)
    }))
  };
};

interface ValidationResultsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  validationResult: ValidationResult | null;
  title?: string;
  showSuccessMessage?: boolean;
}

export const ValidationResultsDialog: React.FC<ValidationResultsDialogProps> = ({
  open,
  onOpenChange,
  validationResult,
  title = "Validation Results",
  showSuccessMessage = true,
}) => {
  if (!validationResult) return null;

  // Parse the validation results to improve error messages
  const parsedResults = parseValidationResults(validationResult);

  const hasNodeIssues = parsedResults.node_results && 
    parsedResults.node_results.some(nr => nr.errors.length > 0 || nr.warnings.length > 0);

  const nodeErrorCount = parsedResults.node_results?.reduce(
    (sum, nr) => sum + nr.errors.length, 0
  ) || 0;

  const nodeWarningCount = parsedResults.node_results?.reduce(
    (sum, nr) => sum + nr.warnings.length, 0
  ) || 0;

  const totalErrors = parsedResults.errors.length + nodeErrorCount;
  const totalWarnings = parsedResults.warnings.length + nodeWarningCount;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            {parsedResults.valid ? (
              <CheckCircle className="h-5 w-5 text-green-500" />
            ) : (
              <AlertCircle className="h-5 w-5 text-red-500" />
            )}
            <span>{title}</span>
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4">
          {/* Success Summary */}
          {parsedResults.valid && showSuccessMessage && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-md">
              <div className="flex items-center space-x-2 mb-2">
                <CheckCircle className="h-5 w-5 text-green-500" />
                <h4 className="font-medium text-green-800">Validation Successful</h4>
              </div>
              <p className="text-sm text-green-700 mb-3">
                Your project structure is valid and ready for use.
              </p>
              
              <div className="grid grid-cols-3 gap-4 text-sm">
                {parsedResults.node_count !== undefined && (
                  <div className="text-gray-600">
                    <span className="font-medium">Nodes:</span> {parsedResults.node_count}
                  </div>
                )}
                {parsedResults.feature_count !== undefined && (
                  <div className="text-gray-600">
                    <span className="font-medium">Features:</span> {parsedResults.feature_count}
                  </div>
                )}
                {parsedResults.subtree_count !== undefined && (
                  <div className="text-gray-600">
                    <span className="font-medium">Subtrees:</span> {parsedResults.subtree_count}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Error Summary */}
          {!parsedResults.valid && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-md">
              <div className="flex items-center space-x-2 mb-2">
                <AlertCircle className="h-5 w-5 text-red-500" />
                <h4 className="font-medium text-red-800">Validation Failed</h4>
              </div>
              <p className="text-sm text-red-700">
                Your project has {totalErrors} error{totalErrors !== 1 ? 's' : ''} 
                {totalWarnings > 0 && ` and ${totalWarnings} warning${totalWarnings !== 1 ? 's' : ''}`} 
                {' '}that need to be addressed.
              </p>
            </div>
          )}

          {/* Tree-Level Issues */}
          {(parsedResults.errors.length > 0 || parsedResults.warnings.length > 0) && (
            <div className="space-y-3">
              <h5 className="font-medium text-gray-900">Tree-Level Issues</h5>
              
              {/* Tree Warnings */}
              {parsedResults.warnings.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center space-x-1 text-yellow-700">
                    <AlertTriangle className="h-4 w-4" />
                    <span className="text-sm font-medium">
                      Warnings ({parsedResults.warnings.length})
                    </span>
                  </div>
                  <div className="bg-yellow-50 border border-yellow-200 rounded-md p-3">
                    <ul className="list-disc list-inside text-sm text-yellow-700 space-y-1">
                      {parsedResults.warnings.map((warning, index) => (
                        <li key={index}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {/* Tree Errors */}
              {parsedResults.errors.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center space-x-1 text-red-700">
                    <AlertCircle className="h-4 w-4" />
                    <span className="text-sm font-medium">
                      Errors ({parsedResults.errors.length})
                    </span>
                  </div>
                  <div className="bg-red-50 border border-red-200 rounded-md p-3">
                    <ul className="list-disc list-inside text-sm text-red-700 space-y-1">
                      {parsedResults.errors.map((error, index) => (
                        <li key={index}>{error}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Node-Level Issues */}
          {hasNodeIssues && (
            <div className="space-y-3">
              <h5 className="font-medium text-gray-900">
                Node-Level Issues ({nodeErrorCount} error{nodeErrorCount !== 1 ? 's' : ''}, {nodeWarningCount} warning{nodeWarningCount !== 1 ? 's' : ''})
              </h5>
              
              <Accordion type="single" collapsible className="w-full">
                {parsedResults.node_results?.map((nodeResult, index) => {
                  const hasIssues = nodeResult.errors.length > 0 || nodeResult.warnings.length > 0;
                  if (!hasIssues) return null;

                  const severity = nodeResult.errors.length > 0 ? 'error' : 'warning';
                  
                  return (
                    <AccordionItem key={index} value={`node-${index}`}>
                      <AccordionTrigger className={cn(
                        "hover:no-underline",
                        severity === 'error' ? "text-red-700" : "text-yellow-700"
                      )}>
                        <div className="flex items-center space-x-2">
                          {severity === 'error' ? (
                            <AlertCircle className="h-4 w-4 text-red-500" />
                          ) : (
                            <AlertTriangle className="h-4 w-4 text-yellow-500" />
                          )}
                          <span className="font-mono text-sm">{nodeResult.node_id}</span>
                          {nodeResult.node_type && (
                            <span className="text-xs text-gray-500">({nodeResult.node_type})</span>
                          )}
                          <span className="text-xs">
                            - {nodeResult.errors.length} error{nodeResult.errors.length !== 1 ? 's' : ''}, 
                            {' '}{nodeResult.warnings.length} warning{nodeResult.warnings.length !== 1 ? 's' : ''}
                          </span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent>
                        <div className="space-y-2 pt-2">
                          {nodeResult.errors.length > 0 && (
                            <div className="bg-red-50 border border-red-200 rounded p-2">
                              <p className="text-xs font-medium text-red-700 mb-1">Errors:</p>
                              <ul className="list-disc list-inside text-sm text-red-600 space-y-1">
                                {nodeResult.errors.map((error, errorIndex) => (
                                  <li key={errorIndex}>{error}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {nodeResult.warnings.length > 0 && (
                            <div className="bg-yellow-50 border border-yellow-200 rounded p-2">
                              <p className="text-xs font-medium text-yellow-700 mb-1">Warnings:</p>
                              <ul className="list-disc list-inside text-sm text-yellow-600 space-y-1">
                                {nodeResult.warnings.map((warning, warningIndex) => (
                                  <li key={warningIndex}>{warning}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  );
                })}
              </Accordion>
            </div>
          )}

          {/* Format Info */}
          {parsedResults.format_version !== undefined && (
            <div className="text-xs text-gray-500 pt-2 border-t">
              Format Version: {parsedResults.format_version}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
