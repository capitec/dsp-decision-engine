import React, { useState } from 'react';
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
import { useToast } from "@/hooks/use-toast";
import {
  listDatasourcesApiV1DatasourcesGet,
  listDatasourceTablesApiV1DatasourcesDatasourceNameTablesGet,
  getTableDetailsApiV1DatasourcesDatasourceNameTablesTableNameGet
} from "@/api";
import type { DataSourceInfo } from "@/api/types.gen";
import type { PolarsDataType } from '../flat-rules/output-types';

type InputSchemaField = [string, PolarsDataType];

interface LoadFromDatasetDialogProps {
  open: boolean;
  onClose: () => void;
  onSchemaLoaded: (schema: InputSchemaField[]) => void;
}

export const LoadFromDatasetDialog: React.FC<LoadFromDatasetDialogProps> = ({
  open,
  onClose,
  onSchemaLoaded
}) => {
  const [datasources, setDatasources] = useState<Record<string, DataSourceInfo>>({});
  const [selectedDatasource, setSelectedDatasource] = useState<string>('');
  const [selectedTable, setSelectedTable] = useState<string>('');
  const [tables, setTables] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingTables, setLoadingTables] = useState(false);
  const { toast } = useToast();

  React.useEffect(() => {
    if (open) {
      fetchDatasources();
    }
  }, [open]);

  React.useEffect(() => {
    if (selectedDatasource) {
      fetchTables(selectedDatasource);
    }
  }, [selectedDatasource]);

  const fetchDatasources = async () => {
    try {
      const response = await listDatasourcesApiV1DatasourcesGet();
      if (response.data) {
        const datasourceRecord: Record<string, DataSourceInfo> = {};
        response.data.datasources.forEach(ds => {
          datasourceRecord[ds.name] = ds;
        });
        setDatasources(datasourceRecord);
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

  const handleLoadSchema = async () => {
    if (!selectedDatasource || !selectedTable) {
      toast({
        title: "Error",
        description: "Please select both datasource and table",
        variant: "destructive"
      });
      return;
    }

    try {
      setLoading(true);
      const response = await getTableDetailsApiV1DatasourcesDatasourceNameTablesTableNameGet({
        path: {
          datasource_name: selectedDatasource,
          table_name: selectedTable
        }
      });

      if (response.data) {
        // table_schema can be either an array of tuples or an object
        let schema: InputSchemaField[];

        if (Array.isArray(response.data.table_schema)) {
          // Already in the correct format: [["field1", "Type1"], ["field2", "Type2"]]
          schema = response.data.table_schema as InputSchemaField[];
        } else {
          // Object format: {"field1": "Type1", "field2": "Type2"}
          schema = Object.entries(response.data.table_schema).map(
            ([name, type]) => [name, type as PolarsDataType]
          );
        }

        onSchemaLoaded(schema);
        onClose();
      }
    } catch (error: any) {
      console.error('Failed to load schema:', error);
      toast({
        title: "Error",
        description: error.message || "Failed to load schema from table",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Load Schema from Dataset</DialogTitle>
          <DialogDescription>
            Select a datasource and table to load its schema as input fields
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Datasource</label>
            <Select value={selectedDatasource} onValueChange={setSelectedDatasource}>
              <SelectTrigger>
                <SelectValue placeholder="Select datasource" />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(datasources).map(([name, ds]) => (
                  <SelectItem key={name} value={name}>
                    {ds.display_name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Table</label>
            <Select
              value={selectedTable}
              onValueChange={setSelectedTable}
              disabled={!selectedDatasource || loadingTables}
            >
              <SelectTrigger>
                <SelectValue placeholder={loadingTables ? "Loading tables..." : "Select table"} />
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
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleLoadSchema} disabled={loading || !selectedTable}>
            {loading ? "Loading..." : "Load Schema"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
