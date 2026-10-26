import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { PolarsType, PolarsPrimitiveType, PolarsSimpleComplexType } from '@/types/datasource-types';

interface CreateFeatureDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  initialName: string;
  onConfirm: (name: string, type: PolarsPrimitiveType | PolarsSimpleComplexType) => void;
}

/**
 * Dialog for creating a new feature with name and Polars type.
 *
 * Integrates with the feature_schema (input schema) to add properly typed features.
 */
export function CreateFeatureDialog({
  open,
  onOpenChange,
  initialName,
  onConfirm,
}: CreateFeatureDialogProps) {
  const [name, setName] = useState(initialName);
  const [type, setType] = useState<PolarsPrimitiveType | PolarsSimpleComplexType>('String');

  // Reset state when dialog opens
  React.useEffect(() => {
    if (open) {
      setName(initialName);
      setType('String');
    }
  }, [open, initialName]);

  const handleConfirm = () => {
    if (!name.trim()) return;
    onConfirm(name.trim(), type);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Create New Feature</DialogTitle>
          <DialogDescription>
            Add a new feature to the input schema with a name and Polars type.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="feature-name">Feature Name</Label>
            <Input
              id="feature-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. customer_age"
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  handleConfirm();
                }
              }}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="feature-type">Polars Type</Label>
            <Select value={type} onValueChange={(v) => setType(v as PolarsPrimitiveType | PolarsSimpleComplexType)}>
              <SelectTrigger id="feature-type">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="String">String</SelectItem>
                <SelectItem value="Float64">Float64</SelectItem>
                <SelectItem value="Int64">Int64</SelectItem>
                <SelectItem value="Boolean">Boolean</SelectItem>
                <SelectItem value="Date">Date</SelectItem>
                <SelectItem value="Datetime">Datetime</SelectItem>
                <SelectItem value="Time">Time</SelectItem>
                <SelectItem value="Duration">Duration</SelectItem>
                <SelectItem value="Categorical">Categorical</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleConfirm} disabled={!name.trim()}>
            Create Feature
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
