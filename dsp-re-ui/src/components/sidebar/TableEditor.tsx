// TableEditor.tsx
import React from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { 
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow 
} from "@/components/ui/table";
import { Plus, X } from 'lucide-react';
import type { TreeOutput, TableEditorProps } from './types';

export const TableEditor: React.FC<TableEditorProps> = ({ outputs, onOutputsChange }) => {
  const columns = outputs.length > 0 ? Object.keys(outputs[0]) : [];

  const handleAddColumn = () => {
    const newColumnName = `Column${columns.length + 1}`;
    const newOutputs = outputs.map(row => ({
      ...row,
      [newColumnName]: ''
    }));
    onOutputsChange(newOutputs);
  };

  const handleRemoveColumn = (columnName: string) => {
    const newOutputs = outputs.map(row => {
      const { [columnName]: removed, ...rest } = row;
      return rest;
    });
    onOutputsChange(newOutputs);
  };

  const handleAddRow = () => {
    const newRow = columns.reduce((acc, col) => ({
      ...acc,
      [col]: ''
    }), {});
    onOutputsChange([...outputs, newRow]);
  };

  const handleRemoveRow = (index: number) => {
    const newOutputs = outputs.filter((_, i) => i !== index);
    onOutputsChange(newOutputs);
  };

  const handleCellChange = (rowIndex: number, column: string, value: string) => {
    const newOutputs = [...outputs];
    newOutputs[rowIndex] = {
      ...newOutputs[rowIndex],
      [column]: value
    };
    onOutputsChange(newOutputs);
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h4 className="font-medium">Output Configuration</h4>
        <Button
          variant="outline"
          size="sm"
          onClick={handleAddColumn}
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Column
        </Button>
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            {columns.map(column => (
              <TableHead key={column} className="relative">
                <div className="flex items-center gap-2">
                  {column}
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-4 w-4"
                    onClick={() => handleRemoveColumn(column)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {outputs.map((row, rowIndex) => (
            <TableRow key={rowIndex}>
              {columns.map(column => (
                <TableCell key={`${rowIndex}-${column}`}>
                  <Input
                    value={row[column]}
                    onChange={e => handleCellChange(rowIndex, column, e.target.value)}
                    className="w-full"
                  />
                </TableCell>
              ))}
              <TableCell>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => handleRemoveRow(rowIndex)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      <Button
        variant="outline"
        onClick={handleAddRow}
        className="w-full"
      >
        <Plus className="h-4 w-4 mr-2" />
        Add Row
      </Button>
    </div>
  );
};