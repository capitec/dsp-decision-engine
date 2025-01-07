// TableEditor.tsx
import React, { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
    Popover,
    PopoverContent,
    PopoverTrigger,
} from "@/components/ui/popover";
import { Plus, X } from "lucide-react";
import { useTreeOutput } from "@components/editor/EditorContext";
import { Label } from "@/components/ui/label"


export const TableEditor: React.FC<{}> = () => {
  const [outputs, setOutputs] = useTreeOutput();
  const [newColName, setNewColName] = useState("");
  const [newColDialogOpen, setNewColDialogOpen] = useState(false);
//   const columns = outputs.length > 0 ? Object.keys(outputs[0]) : [];

  const handleAddColumn = useCallback(() => {
    setOutputs(outputs => {
        return {
            columns: [...outputs.columns, newColName],
            data: outputs.data.map(v=>[...v,""])
        }
    })
    setNewColName("");
    setNewColDialogOpen(false);
  }, [newColName]);

  const handleAddRow = useCallback(() => {
    setOutputs(outputs => {
        return {
            columns: outputs.columns,
            data: [...outputs.data, outputs.columns.map(v=>"")]
        }
    })
  }, []);
  

  const handleRemoveColumn = useCallback((columnIdx: number) => {
    setOutputs((outputs) => {
      if (columnIdx < 0 || columnIdx >= outputs.columns.length) return outputs;
      const newColumns = [...outputs.columns];
      newColumns.splice(columnIdx, 1);
      const newData = outputs.data.map((row) => {
        const newRow = [...row];
        newRow.splice(columnIdx, 1);
        return newRow;
      });
      return {
        columns: newColumns,
        data: newData,
      };
    });
  }, []);


  const handleRenameColumn = useCallback((columnIdx: number, newValue: string) => {
    setOutputs((outputs) => {
      if (columnIdx < 0 || columnIdx >= outputs.columns.length) return outputs;
      const newColumns = [...outputs.columns];
      newColumns[columnIdx] = newValue;
      return {
        columns: newColumns,
        data: outputs.data,
      };
    });
  }, []);


  const handleRemoveRow = useCallback((index: number) => {
    setOutputs((outputs) => {
      if (index < 0 || index >= outputs.data.length) return outputs;
      return {
        columns: outputs.columns,
        data: outputs.data.filter((_, idx) => idx !== index),
      };
    });
  }, []);

  const handleCellChange = useCallback((
    rowIndex: number,
    columnIndex: number,
    value: string
  ) => {
    setOutputs((outputs) => {
        if (rowIndex < 0 || rowIndex >= outputs.data.length) return outputs;
        if (columnIndex < 0 || columnIndex >= outputs.columns.length) return outputs;
        const newData = [...outputs.data];
        newData[rowIndex] = [...newData[rowIndex]];
        newData[rowIndex][columnIndex] = value;
      return {
        columns: outputs.columns,
        data: newData,
      };
    });
  }, []);



  return (
    <div className="space-y-4">

      <Table>
        <TableHeader>
          <TableRow>
            
            <TableHead key="_id" className="relative">
                <div className="flex items-center gap-2">
                    id
                </div>
            </TableHead>
            {outputs.columns.map((column, colIndex) => (
              <TableHead key={`col-${colIndex}`} className="relative">
                <div className="flex items-center gap-2">
                  <Input
                    value={column}
                    onChange={(e) =>
                      handleRenameColumn(colIndex, e.target.value)
                    }
                    className="w-full"
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-4 w-4"
                    onClick={() => handleRemoveColumn(colIndex)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              </TableHead>
            ))}

            <TableHead key="_add_column" className="relative">
                <div className="flex items-center gap-2">
                <Popover onOpenChange={setNewColDialogOpen} open={newColDialogOpen}>
                    <PopoverTrigger asChild>
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-4 w-4"
                        >
                            <Plus className="h-3 w-3" />
                        </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-80">
                        <Label htmlFor="new-col-name">Column Name</Label>
                        <Input
                            value={newColName}
                            id="new-col-name"
                            placeholder="Column Name"
                            onChange={(e)=>setNewColName(e.target.value)}
                            className="w-full"
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                    handleAddColumn(); 
                                }
                            }}
                        />
                        <Button variant="outline" size="sm" onClick={handleAddColumn} className="mt-[10px]">
                        <Plus className="h-4 w-4 mr-2" />
                        Add Column
                        </Button>
                    </PopoverContent>
                </Popover>
                  {/* <Input
                    value={newColName}
                    placeholder="Column Name"
                    onChange={(e)=>setNewColName(e.target.value)}
                    className="w-[30px]"
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-4 w-4"
                    onClick={handleAddColumn}
                  >
                    <Plus className="h-3 w-3" />
                  </Button> */}
                </div>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {outputs.data.map((row, rowIndex) => (
            <TableRow key={rowIndex}>

            <TableCell key={`_id_${rowIndex}`}>
                {rowIndex}
            </TableCell>
              {row.map((value, colIndex) => (
                <TableCell key={`${rowIndex}-${colIndex}`}>
                  <Input
                    value={value}
                    onChange={(e) =>
                      handleCellChange(rowIndex, colIndex, e.target.value)
                    }
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

      <Button variant="outline" onClick={handleAddRow} className="w-full">
        <Plus className="h-4 w-4 mr-2" />
        Add Row
      </Button>
    </div>
  );
};
