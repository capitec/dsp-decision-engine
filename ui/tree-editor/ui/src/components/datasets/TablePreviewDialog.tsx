import React, { useState, useEffect, useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { 
  RefreshCw, 
  Database, 
  Calendar, 
  HardDrive, 
  Users,
  X 
} from 'lucide-react';
import {
  getTableDetailsApiV1DatasourcesDatasourceNameTablesTableNameGet,
  previewTableDataApiV1DatasourcesDatasourceNameTablesTableNamePreviewGet
} from "@/api";
import { 
  TableInfo, 
  TableDetailResponse, 
  TablePreviewResponse 
} from '../../types/datasource-types';

interface TablePreviewDialogProps {
  datasourceName: string;
  tableName: string;
  onClose: () => void;
}

export const TablePreviewDialog: React.FC<TablePreviewDialogProps> = ({
  datasourceName,
  tableName,
  onClose
}) => {
  const [tableDetails, setTableDetails] = useState<TableDetailResponse | null>(null);
  const [previewData, setPreviewData] = useState<TablePreviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'preview' | 'schema'>('preview');
  const [error, setError] = useState<string | null>(null);

  // Using this to prevent a double fetch
  const tableDsCompoundKey = useMemo(() => `${datasourceName}.${tableName}`, [datasourceName, tableName]);

  useEffect(() => {
    loadTableData();
  }, [tableDsCompoundKey]);

  const loadTableData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load both table details and preview data
      const [detailsResponse, previewResponse] = await Promise.all([
        getTableDetailsApiV1DatasourcesDatasourceNameTablesTableNameGet({
          path: {
            datasource_name: datasourceName,
            table_name: tableName
          }
        }),
        previewTableDataApiV1DatasourcesDatasourceNameTablesTableNamePreviewGet({
          path: {
            datasource_name: datasourceName,
            table_name: tableName
          },
          query: { limit: 20 }
        })
      ]);

      if (detailsResponse.data) setTableDetails(detailsResponse.data as TableDetailResponse); // Type mismatch between API TStructOutput and PolarsSchema
      if (previewResponse.data) setPreviewData(previewResponse.data);
    } catch (error: any) {
      console.error('Failed to load table data:', error);
      setError(error.message || 'Failed to load table data');
    } finally {
      setLoading(false);
    }
  };

  const formatBytes = (bytes: number) => {
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat().format(num);
  };

  const renderCellValue = (value: any): string => {
    if (value === null || value === undefined) {
      return 'null';
    }
    if (typeof value === 'object') {
      return JSON.stringify(value);
    }
    return String(value);
  };

  const getTypeDisplayName = (type: any): string => {
    if (typeof type === 'string') {
      return type;
    }
    if (typeof type === 'object' && type !== null && type.type) {
      // Explicit type like {"type": "Datetime", "time_unit": "us"}
      const typeName = type.type;
      
      // Handle specific type formatting
      if (typeName === 'Datetime') {
        const parts = [typeName];
        if (type.time_unit) parts.push(`(${type.time_unit})`);
        if (type.time_zone) parts.push(`[${type.time_zone}]`);
        return parts.join('');
      }
      
      if (typeName === 'Decimal') {
        return `Decimal(${type.precision || '?'},${type.scale})`;
      }
      
      if (typeName === 'Duration') {
        return type.time_unit ? `Duration(${type.time_unit})` : 'Duration';
      }
      
      if (typeName === 'List' || typeName === 'Array') {
        const innerType = typeof type.inner === 'string' ? type.inner : getTypeDisplayName(type.inner);
        return typeName === 'Array' ? `Array<${innerType}>[${type.width}]` : `List<${innerType}>`;
      }
      
      if (typeName === 'Struct') {
        return 'Struct';
      }
      
      if (typeName === 'Categorical' || typeName === 'Enum') {
        return typeName;
      }
      
      return typeName;
    }
    return String(type);
  };

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            <div className="flex items-center">
              <Database className="w-5 h-5 mr-2" />
              {tableName}
              <Badge className="ml-2" variant="outline">
                {datasourceName}
              </Badge>
            </div>
            {/* <Button onClick={onClose} variant="ghost" size="sm">
              <X className="w-4 h-4" />
            </Button> */}
          </DialogTitle>
          <DialogDescription>
            View table schema, preview data, and explore the structure of {tableName} from the {datasourceName} datasource.
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center space-x-2">
              <RefreshCw className="h-4 w-4 animate-spin" />
              <span>Loading table data...</span>
            </div>
          </div>
        ) : error ? (
          <div className="text-center py-12">
            <div className="text-red-600 mb-2">Failed to load table data</div>
            <div className="text-sm text-gray-500 mb-4">{error}</div>
            <Button onClick={loadTableData} variant="outline">
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry
            </Button>
          </div>
        ) : (
          <div className="space-y-4 flex-1 overflow-auto">
            {/* Table Info */}
            {tableDetails && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-900">
                    {tableDetails.info.row_count ? formatNumber(tableDetails.info.row_count) : 'Unknown'}
                  </div>
                  <div className="text-sm text-gray-600">Rows</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-900">
                    {previewData ? previewData.data.total_columns : tableDetails.table_schema.length}
                  </div>
                  <div className="text-sm text-gray-600">Columns</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-900">
                    {tableDetails.info.size_bytes ? formatBytes(tableDetails.info.size_bytes) : 'Unknown'}
                  </div>
                  <div className="text-sm text-gray-600">Size</div>
                </div>
                <div className="text-center">
                  <div className="text-sm font-medium text-gray-900">
                    {tableDetails.info.last_modified 
                      ? new Date(tableDetails.info.last_modified).toLocaleDateString()
                      : 'Unknown'
                    }
                  </div>
                  <div className="text-sm text-gray-600">Last Modified</div>
                </div>
              </div>
            )}

            {/* Tabs */}
            <div className="flex border-b">
              <button
                onClick={() => setActiveTab('preview')}
                className={`px-4 py-2 text-sm font-medium border-b-2 ${
                  activeTab === 'preview'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                Data Preview
                {previewData && (
                  <Badge className="ml-2" variant="outline">
                    {previewData.data.row_count} rows shown
                  </Badge>
                )}
              </button>
              <button
                onClick={() => setActiveTab('schema')}
                className={`px-4 py-2 text-sm font-medium border-b-2 ${
                  activeTab === 'schema'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                Schema
                {tableDetails && (
                  <Badge className="ml-2" variant="outline">
                    {tableDetails.table_schema.length} columns
                  </Badge>
                )}
              </button>
            </div>

            {/* Tab Content */}
            <div className="h-96 overflow-x-auto overflow-y-auto border rounded-md">
              {activeTab === 'preview' && previewData ? (
                <div>
                  <table className="caption-bottom text-sm min-w-max">
                    <thead className="[&_tr]:border-b">
                      <tr className="border-b transition-colors hover:bg-muted/50">
                        {previewData.data.columns.map((column) => (
                          <th key={column} className="h-12 px-4 text-left align-middle font-medium text-muted-foreground whitespace-nowrap">
                            {column}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="[&_tr:last-child]:border-0">
                      {previewData.data.rows.map((row, rowIndex) => (
                        <tr key={rowIndex} className="border-b transition-colors hover:bg-muted/50">
                          {row.map((cell, cellIndex) => (
                            <td key={cellIndex} className="p-4 align-middle whitespace-nowrap min-w-[100px] max-w-md">
                              <span className={cell === null || cell === undefined ? 'text-gray-400 italic' : ''}>
                                {renderCellValue(cell)}
                              </span>
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : activeTab === 'schema' && tableDetails ? (
                <div>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Column Name</TableHead>
                        <TableHead>Data Type</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {tableDetails.table_schema.map(([columnName, columnType]) => (
                        <TableRow key={columnName}>
                          <TableCell className="font-medium">{columnName}</TableCell>
                          <TableCell>
                            <Badge variant="outline">
                              {getTypeDisplayName(columnType)}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : null}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};