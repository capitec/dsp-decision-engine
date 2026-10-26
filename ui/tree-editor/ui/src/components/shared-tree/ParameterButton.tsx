import React, { useState, useCallback } from 'react';
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter
} from "@/components/ui/dialog";
import { FlatRuleParameter } from './ParameterTypes';

export type { FlatRuleParameter } from './ParameterTypes';

export interface ParameterValue {
  key: string;
}
interface ParameterButtonProps {
  value: string | number;                    // Current value
  onChange: (value: string | number) => void; // Callback when parameter is selected
  parameters?: FlatRuleParameter[];         // Available parameters
  compatibleParameterTypes?: string[];     // Parameter types that are compatible
  disabled?: boolean;
}

export const ParameterButton: React.FC<ParameterButtonProps> = ({
  value,
  onChange,
  parameters = [],
  compatibleParameterTypes = [],
  disabled = false
}) => {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [selectedParameter, setSelectedParameter] = useState<string | null>(null);
  
  // Check if current value is a parameter reference
  const isParameterReference = typeof value === 'string' && value.startsWith('#');
  
  // Filter parameters by compatible types
  const filteredParameters = compatibleParameterTypes.length === 0 ?
    parameters :
    parameters.filter(p => {
      const typeStr = typeof p.parameter_type === 'string' ? p.parameter_type : 'String';
      return compatibleParameterTypes.includes(typeStr);
    });


  const handleParameterSelect = useCallback(() => {
    if (selectedParameter) {
      onChange(`#${selectedParameter}`);
      setIsDialogOpen(false);
      setSelectedParameter(null);
    }
  }, [selectedParameter, onChange]);

  const handleClearParameter = useCallback(() => {
    // Reset to default value based on type
    const defaultValue = compatibleParameterTypes.includes('number') ? 0 : '';
    onChange(defaultValue);
    setIsDialogOpen(false);
    setSelectedParameter(null);
  }, [onChange, compatibleParameterTypes]);

  // Get the current parameter info if it's a parameter reference
  const currentParameter = isParameterReference ? 
    parameters.find(p => p.name === (value as string).slice(1)) : 
    null;

  return (
    <>
      <Button
        type="button"
        variant={isParameterReference ? "default" : "outline"}
        size="sm"
        className="h-8 w-8 p-0 text-xs flex-shrink-0"
        onClick={() => setIsDialogOpen(true)}
        disabled={disabled || filteredParameters.length === 0}
        title={
          filteredParameters.length === 0 
            ? "No compatible parameters available" 
            : isParameterReference 
              ? `Using parameter: ${currentParameter?.name || 'unknown'}`
              : "Set parameter reference"
        }
      >
        #
      </Button>

      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Set Parameter Reference</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4">
            {isParameterReference && currentParameter && (
              <div className="p-3 bg-blue-50 rounded border">
                <div className="text-sm font-medium">Currently using parameter:</div>
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant="secondary">{currentParameter.name}</Badge>
                  <span className="text-sm text-gray-600">= {String(currentParameter.value)}</span>
                </div>
              </div>
            )}
            
            <div className="space-y-2">
              <label className="text-sm font-medium">Select Parameter:</label>
              <Select value={selectedParameter || ''} onValueChange={setSelectedParameter}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose a parameter..." />
                </SelectTrigger>
                <SelectContent>
                  {filteredParameters.map(parameter => {
                    const typeStr = typeof parameter.parameter_type === 'string'
                      ? parameter.parameter_type
                      : 'Complex';
                    return (
                      <SelectItem key={parameter.id} value={parameter.name}>
                        <div className="flex items-center justify-between w-full">
                          <span>{parameter.name}</span>
                          <div className="flex items-center gap-2 ml-2">
                            <Badge variant="secondary" className="text-xs">
                              {typeStr}
                            </Badge>
                            <span className="text-xs text-gray-500">
                              = {String(parameter.value)}
                            </span>
                          </div>
                        </div>
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>
          </div>

          <DialogFooter className="gap-2">
            {isParameterReference && (
              <Button 
                variant="outline" 
                onClick={handleClearParameter}
                className="mr-auto"
              >
                Clear Parameter
              </Button>
            )}
            <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
              Cancel
            </Button>
            <Button 
              onClick={handleParameterSelect}
              disabled={!selectedParameter}
            >
              Set Parameter
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};