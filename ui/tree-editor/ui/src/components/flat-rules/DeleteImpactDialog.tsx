import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { AlertTriangle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import type { CategoryReference, StructRecordReference } from './referenceAnalysis';

interface DeleteImpactDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;

  // What's being deleted
  itemType: 'category' | 'struct-record';
  itemName: string;
  typeName: string;

  // References found
  categoryReferences?: CategoryReference[];
  structReferences?: StructRecordReference[];

  // Swaps (for struct types)
  swaps?: Array<{ from: string; to: string }>;
}

export const DeleteImpactDialog: React.FC<DeleteImpactDialogProps> = ({
  open,
  onOpenChange,
  onConfirm,
  itemType,
  itemName,
  typeName,
  categoryReferences = [],
  structReferences = [],
  swaps = []
}) => {
  const references = itemType === 'category' ? categoryReferences : structReferences;
  const hasReferences = references.length > 0;
  const hasSwaps = swaps.length > 0;

  // Group references by field name
  const groupedByField = references.reduce((acc, ref) => {
    if (!acc[ref.fieldName]) {
      acc[ref.fieldName] = { single: 0, array: 0, recordIndices: [] };
    }
    if (ref.inArray) {
      acc[ref.fieldName].array++;
    } else {
      acc[ref.fieldName].single++;
    }
    acc[ref.fieldName].recordIndices.push(ref.recordIndex);
    return acc;
  }, {} as Record<string, { single: number; array: number; recordIndices: number[] }>);

  const handleConfirm = () => {
    onConfirm();
    onOpenChange(false);
  };

  const handleCancel = () => {
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-orange-500" />
            {hasReferences && !hasSwaps && `Delete ${itemType === 'category' ? 'Category' : 'Record'}?`}
            {hasSwaps && !hasReferences && 'Update References?'}
            {hasSwaps && hasReferences && 'Confirm Changes?'}
          </DialogTitle>
          <DialogDescription>
            {!hasSwaps && (
              <>You are about to delete <strong>{itemName}</strong> from <strong>{typeName}</strong></>
            )}
            {hasSwaps && !hasReferences && (
              <>You are about to update <strong>{swaps.length}</strong> display value{swaps.length !== 1 ? 's' : ''} in <strong>{typeName}</strong></>
            )}
            {hasSwaps && hasReferences && (
              <>You are about to make changes to <strong>{typeName}</strong> that affect output data</>
            )}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Show swap information */}
          {hasSwaps && (
            <Alert>
              <AlertDescription>
                <p className="font-medium mb-2">
                  {swaps.length} display value{swaps.length !== 1 ? 's' : ''} will be updated:
                </p>
                <ul className="list-disc list-inside space-y-1 text-sm">
                  {swaps.map((swap, idx) => (
                    <li key={idx}>
                      <strong>{swap.from}</strong> → <strong>{swap.to}</strong>
                    </li>
                  ))}
                </ul>
                <p className="text-sm text-gray-600 mt-2">
                  All references to these values in output data will be automatically updated.
                </p>
              </AlertDescription>
            </Alert>
          )}

          {hasReferences ? (
            <>
              <Alert variant="destructive">
                <AlertDescription>
                  <p className="font-medium mb-2">
                    This {itemType === 'category' ? 'category' : 'record'} is currently used in {references.length} location{references.length !== 1 ? 's' : ''}:
                  </p>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    {Object.entries(groupedByField).map(([fieldName, counts]) => {
                      const parts: string[] = [];
                      if (counts.single > 0) {
                        parts.push(`${counts.single} single value${counts.single !== 1 ? 's' : ''}`);
                      }
                      if (counts.array > 0) {
                        parts.push(`${counts.array} array${counts.array !== 1 ? 's' : ''}`);
                      }
                      const recordSummary = counts.recordIndices.length <= 3
                        ? `record${counts.recordIndices.length !== 1 ? 's' : ''} ${counts.recordIndices.map(i => i + 1).join(', ')}`
                        : `${counts.recordIndices.length} records`;

                      return (
                        <li key={fieldName}>
                          <strong>{fieldName}</strong>: {parts.join(' and ')} ({recordSummary})
                        </li>
                      );
                    })}
                  </ul>
                </AlertDescription>
              </Alert>

              <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded border">
                {itemType === 'category' ? (
                  <p>
                    Deleting this category will:
                    <ul className="list-disc list-inside mt-1 ml-2">
                      <li>Set single values to <strong>empty string</strong></li>
                      <li>Remove from all lists/sets/arrays</li>
                    </ul>
                  </p>
                ) : (
                  <p>
                    Deleting this record will:
                    <ul className="list-disc list-inside mt-1 ml-2">
                      <li>Set single references to <strong>null</strong></li>
                      <li>Remove from all lists/sets/arrays</li>
                    </ul>
                  </p>
                )}
              </div>
            </>
          ) : (
            <div className="text-sm text-gray-600">
              This {itemType === 'category' ? 'category' : 'record'} is not currently used in any output records.
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button
            variant={hasReferences && !hasSwaps ? "destructive" : "default"}
            onClick={handleConfirm}
          >
            {hasSwaps && !hasReferences && 'Confirm Updates'}
            {hasReferences && !hasSwaps && 'Delete and Clean Up'}
            {hasSwaps && hasReferences && 'Confirm All Changes'}
            {!hasSwaps && !hasReferences && 'Delete'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
