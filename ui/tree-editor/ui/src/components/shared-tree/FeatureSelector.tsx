import React, { useState, useMemo, useCallback } from 'react';
import { Label } from "@/components/ui/label";
import {
  Command,
  CommandList,
  CommandGroup,
  CommandInput,
  CommandItem,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { CreateFeatureDialog } from './CreateFeatureDialog';
import { useEditorStore } from '../editor/useEditorStore';
import { PolarsPrimitiveType, PolarsSimpleComplexType } from '@/types/datasource-types';

interface FeatureSelectorProps {
  /** Current feature name (empty string if none selected) */
  value: string;
  /** Callback when feature is selected or created */
  onChange: (featureName: string) => void;
  /** List of available feature names (derived from feature_schema) */
  availableFeatures: string[];
  /** Label to display above the selector */
  label?: string;
  /** Placeholder when no feature is selected */
  placeholder?: string;
  /** Allow creating new features on-the-fly (auto-added to feature_schema) */
  allowCreate?: boolean;
  className?: string;
}

/**
 * Searchable feature selector with optional on-the-fly creation.
 *
 * Shared between tree editor and flat rules.
 * Accepts feature names as strings (v2 format).
 */
export function FeatureSelector({
  value,
  onChange,
  availableFeatures,
  label = "Feature",
  placeholder = "Select feature...",
  allowCreate = true,
  className,
}: FeatureSelectorProps) {
  const [open, setOpen] = useState(false);
  const [searchValue, setSearchValue] = useState('');
  const [dialogOpen, setDialogOpen] = useState(false);
  const [pendingFeatureName, setPendingFeatureName] = useState('');

  const setFeatureSchema = useEditorStore((state) => state.setInputSchema);
  const featureSchema = useEditorStore((state) => state.input_schema);

  const showCreate = useMemo(() => {
    if (!searchValue.trim() || !allowCreate) return false;
    // Don't show create if already exists (case-sensitive)
    return !availableFeatures.includes(searchValue);
  }, [searchValue, availableFeatures, allowCreate]);

  const handleSelect = useCallback((featureName: string) => {
    if (!featureName) return;
    onChange(featureName);
    setOpen(false);
    setSearchValue('');
  }, [onChange]);

  const handleCreateClick = useCallback(() => {
    setPendingFeatureName(searchValue);
    setOpen(false);
    setDialogOpen(true);
  }, [searchValue]);

  const handleCreateConfirm = useCallback((name: string, type: PolarsPrimitiveType | PolarsSimpleComplexType) => {
    // Add to feature schema
    const newFeature: [string, PolarsPrimitiveType | PolarsSimpleComplexType] = [name, type];
    setFeatureSchema(prev => [...prev, newFeature]);

    // Select the newly created feature
    onChange(name);
    setSearchValue('');
  }, [featureSchema, setFeatureSchema, onChange]);

  return (
    <>
      <div className={cn("space-y-2", className)}>
        {label && <Label>{label}</Label>}
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button
              variant="outline"
              className="w-full justify-start font-normal"
              role="combobox"
              aria-expanded={open}
            >
              {value || placeholder}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-full p-0" align="start">
            <Command>
              <CommandInput
                placeholder="Search or create feature..."
                value={searchValue}
                onValueChange={setSearchValue}
              />
              <CommandList>
                {showCreate && (
                  <CommandGroup>
                    <CommandItem
                      onSelect={handleCreateClick}
                      className="font-medium text-blue-600"
                    >
                      Create "{searchValue}"
                    </CommandItem>
                  </CommandGroup>
                )}
                <CommandGroup>
                  {availableFeatures.map((feature) => (
                    <CommandItem
                      value={feature}
                      onSelect={handleSelect}
                      key={feature}
                    >
                      <Check
                        className={cn(
                          "mr-2 h-4 w-4",
                          value === feature ? "opacity-100" : "opacity-0"
                        )}
                      />
                      {feature}
                    </CommandItem>
                  ))}
                </CommandGroup>
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
      </div>

      <CreateFeatureDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        initialName={pendingFeatureName}
        onConfirm={handleCreateConfirm}
      />
    </>
  );
}
