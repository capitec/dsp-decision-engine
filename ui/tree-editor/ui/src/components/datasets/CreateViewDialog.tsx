import React, { useState, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Play, Eye } from "lucide-react";
import {
  listDatasourcesApiV1DatasourcesGet,
  listDatasourceTablesApiV1DatasourcesDatasourceNameTablesGet,
} from "@/api";
import type { DataSourceInfo } from "@/api/types.gen";

interface CreateViewDialogProps {
  open: boolean;
  onClose: () => void;
  onViewCreated?: (viewName: string, datasourceName: string) => void;
}

interface PreviewData {
  columns: string[];
  data: Record<string, any[]>;
  row_count: number;
}

export const CreateViewDialog: React.FC<CreateViewDialogProps> = ({
  open,
  onClose,
  onViewCreated
}) => {
  const [sourceDatasources, setSourceDatasources] = useState<Record<string, DataSourceInfo>>({});
  const [destDatasources, setDestDatasources] = useState<Record<string, DataSourceInfo>>({});
  const [selectedSourceDatasource, setSelectedSourceDatasource] = useState<string>('');
  const [selectedDestDatasource, setSelectedDestDatasource] = useState<string>('');
  const [selectedTable, setSelectedTable] = useState<string>('');
  const [viewName, setViewName] = useState<string>('');
  const [sqlQuery, setSqlQuery] = useState<string>('');
  const [tables, setTables] = useState<string[]>([]);
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingTables, setLoadingTables] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    if (open) {
      fetchDatasources();
      // Reset state when dialog opens
      setSelectedSourceDatasource('');
      setSelectedDestDatasource('');
      setSelectedTable('');
      setViewName('');
      setSqlQuery('');
      setPreviewData(null);
    }
  }, [open]);

  useEffect(() => {
    if (selectedSourceDatasource) {
      fetchTables(selectedSourceDatasource);
    }
  }, [selectedSourceDatasource]);

  useEffect(() => {
    if (selectedTable) {
      // Set default SQL query template
      setSqlQuery(`SELECT *\nFROM ${selectedTable}\nLIMIT 100`);
    }
  }, [selectedTable]);

  const fetchDatasources = async () => {
    try {
      const response = await listDatasourcesApiV1DatasourcesGet();
      if (response.data) {
        const sourceRecord: Record<string, DataSourceInfo> = {};
        const destRecord: Record<string, DataSourceInfo> = {};

        response.data.datasources.forEach(ds => {
          // Source: needs discovery capability
          if (ds.capabilities.includes('discovery')) {
            sourceRecord[ds.name] = ds;
          }
          // Destination: needs upload capability
          if (ds.capabilities.includes('upload')) {
            destRecord[ds.name] = ds;
          }
        });

        setSourceDatasources(sourceRecord);
        setDestDatasources(destRecord);
      }
    } catch (error) {
      console.error('Failed to fetch datasources:', error);
      toast({
        title: "Error",
        description: "Failed to load datasources",
        variant: "destructive"
      });
    }
  };

  const fetchTables = async (datasourceName: string) => {
    try {
      setLoadingTables(true);
      const response = await listDatasourceTablesApiV1DatasourcesDatasourceNameTablesGet({
        path: { datasource_name: datasourceName },
        query: { page: 1, page_size: 1000 }
      });

      if (response.data) {
        setTables(response.data.tables || []);
      }
    } catch (error) {
      console.error('Failed to fetch tables:', error);
      toast({
        title: "Error",
        description: "Failed to load tables",
        variant: "destructive"
      });
      setTables([]);
    } finally {
      setLoadingTables(false);
    }
  };

  const handlePreview = async () => {
    if (!selectedSourceDatasource || !selectedTable || !sqlQuery.trim()) {
      toast({
        title: "Error",
        description: "Please select source datasource, table, and provide SQL query",
        variant: "destructive"
      });
      return;
    }

    try {
      setPreviewing(true);
      setPreviewData(null);

      // Create FormData for the API call
      const formData = new FormData();
      formData.append('source_table', selectedTable);
      formData.append('sql_query', sqlQuery);

      const response = await fetch(
        `/api/v1/datasources/${selectedSourceDatasource}/preview-query?limit=20`,
        {
          method: 'POST',
          body: formData,
          credentials: 'include',
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Preview failed');
      }

      const result = await response.json();
      setPreviewData(result.data);

      toast({
        title: "Preview Ready",
        description: `Showing ${result.data.row_count} rows`,
      });
    } catch (error: any) {
      console.error('Failed to preview query:', error);
      toast({
        title: "Preview Failed",
        description: error.message || "Failed to preview SQL query",
        variant: "destructive"
      });
    } finally {
      setPreviewing(false);
    }
  };

  const handleCreateView = async () => {
    if (!selectedSourceDatasource || !selectedDestDatasource || !selectedTable || !viewName.trim() || !sqlQuery.trim()) {
      toast({
        title: "Error",
        description: "Please fill in all fields including source and destination datasources",
        variant: "destructive"
      });
      return;
    }

    // Validate view name (alphanumeric + underscore)
    if (!/^[a-zA-Z0-9_]+$/.test(viewName)) {
      toast({
        title: "Invalid View Name",
        description: "View name must contain only letters, numbers, and underscores",
        variant: "destructive"
      });
      return;
    }

    try {
      setLoading(true);

      // Create view - query from source, save to destination
      const formData = new FormData();
      formData.append('source_datasource', selectedSourceDatasource);
      formData.append('dest_datasource', selectedDestDatasource);
      formData.append('source_table', selectedTable);
      formData.append('view_name', viewName);
      formData.append('sql_query', sqlQuery);

      const createResponse = await fetch(
        `/api/v1/datasources/create-view`,
        {
          method: 'POST',
          body: formData,
          credentials: 'include',
        }
      );

      if (!createResponse.ok) {
        const errorData = await createResponse.json();
        throw new Error(errorData.detail || errorData.message || 'View creation failed');
      }

      const createResult = await createResponse.json();

      if (!createResult.success) {
        throw new Error(createResult.message || 'View creation failed');
      }

      // Notify parent and close
      toast({
        title: "View Created Successfully",
        description: `View "${createResult.table_name}" is now available in ${selectedDestDatasource}`,
      });

      if (onViewCreated) {
        onViewCreated(createResult.table_name, selectedDestDatasource);
      }

      onClose();
    } catch (error: any) {
      console.error('Failed to create view:', error);
      toast({
        title: "Error",
        description: error.message || "Failed to create view from SQL query",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Create View from SQL Query</DialogTitle>
          <DialogDescription>
            Execute a SQL query against a table and save the result as a new view
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4 flex-1 overflow-y-auto">
          {/* Source datasource selector */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Source Datasource (Read From)</label>
            <Select value={selectedSourceDatasource} onValueChange={setSelectedSourceDatasource}>
              <SelectTrigger>
                <SelectValue placeholder="Select source datasource" />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(sourceDatasources).map(([name, ds]) => (
                  <SelectItem key={name} value={name}>
                    {ds.display_name} ({ds.provider})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Source table selector */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Source Table</label>
            <Select
              value={selectedTable}
              onValueChange={setSelectedTable}
              disabled={!selectedSourceDatasource || loadingTables}
            >
              <SelectTrigger>
                <SelectValue placeholder={loadingTables ? "Loading tables..." : "Select source table"} />
              </SelectTrigger>
              <SelectContent>
                {tables.map((table) => (
                  <SelectItem key={table} value={table}>
                    {table}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Destination datasource selector */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Destination Datasource (Save To)</label>
            <Select value={selectedDestDatasource} onValueChange={setSelectedDestDatasource}>
              <SelectTrigger>
                <SelectValue placeholder="Select destination datasource" />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(destDatasources).map(([name, ds]) => (
                  <SelectItem key={name} value={name}>
                    {ds.display_name} ({ds.provider})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* View name input */}
          <div className="space-y-2">
            <label className="text-sm font-medium">View Name</label>
            <Input
              placeholder="my_filtered_view"
              value={viewName}
              onChange={(e) => setViewName(e.target.value)}
              maxLength={64}
            />
            <p className="text-xs text-muted-foreground">
              Letters, numbers, and underscores only
            </p>
          </div>

          {/* SQL Editor */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">SQL Query</label>
              <Button
                onClick={handlePreview}
                disabled={!selectedTable || !sqlQuery.trim() || previewing}
                variant="outline"
                size="sm"
              >
                {previewing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Previewing...
                  </>
                ) : (
                  <>
                    <Eye className="w-4 h-4 mr-2" />
                    Preview
                  </>
                )}
              </Button>
            </div>
            <div className="border rounded-md overflow-hidden">
              <Editor
                height="200px"
                defaultLanguage="sql"
                theme="vs-dark"
                value={sqlQuery}
                onChange={(value) => setSqlQuery(value || '')}
                options={{
                  minimap: { enabled: false },
                  fontSize: 13,
                  lineNumbers: 'on',
                  scrollBeyondLastLine: false,
                  wordWrap: 'on',
                }}
              />
            </div>
          </div>

          {/* Preview results */}
          {previewData && (
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Preview Results ({previewData.row_count} rows)
              </label>
              <div className="border rounded-md overflow-auto max-h-64">
                <table className="w-full text-sm">
                  <thead className="bg-muted">
                    <tr>
                      {previewData.columns.map((col) => (
                        <th key={col} className="px-4 py-2 text-left font-medium">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Array.from({ length: previewData.row_count }).map((_, rowIdx) => (
                      <tr key={rowIdx} className="border-t">
                        {previewData.columns.map((col) => (
                          <td key={col} className="px-4 py-2">
                            {String(previewData.data[col][rowIdx])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={loading}>
            Cancel
          </Button>
          <Button
            onClick={handleCreateView}
            disabled={loading || !selectedSourceDatasource || !selectedDestDatasource || !selectedTable || !viewName.trim() || !sqlQuery.trim()}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Create View
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
