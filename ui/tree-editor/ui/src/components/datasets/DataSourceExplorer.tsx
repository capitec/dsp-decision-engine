import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { 
  Search, 
  ChevronRight, 
  ChevronLeft, 
  Eye, 
  Table, 
  RefreshCw,
  FileText,
  Calendar,
  HardDrive
} from 'lucide-react';
import { TablePreviewDialog } from './TablePreviewDialog';
import {
  DataSourceInfo,
  DataSourceTablesResponse,
  DataSourceSearchResponse
} from '../../types/datasource-types';
import {
  listDatasourceTablesApiV1DatasourcesDatasourceNameTablesGet,
  searchDatasourceTablesApiV1DatasourcesDatasourceNameTablesSearchGet
} from "@/api";

interface DataSourceExplorerProps {
  datasourceName: string;
  datasourceInfo: DataSourceInfo;
}

export const DataSourceExplorer: React.FC<DataSourceExplorerProps> = ({
  datasourceName,
  datasourceInfo
}) => {
  const [tables, setTables] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [pagination, setPagination] = useState({
    page: 1,
    page_size: 20,
    total_count: 0,
    has_next: false,
    has_previous: false
  });
  const [previewTable, setPreviewTable] = useState<string | null>(null);
  const [isSearchMode, setIsSearchMode] = useState(false);

  useEffect(() => {
    loadTables();
  }, [datasourceName, pagination.page]);

  const loadTables = async () => {
    try {
      setLoading(true);

      let data: DataSourceTablesResponse | DataSourceSearchResponse | undefined;

      if (isSearchMode && searchQuery.trim()) {
        const response = await searchDatasourceTablesApiV1DatasourcesDatasourceNameTablesSearchGet({
          path: { datasource_name: datasourceName },
          query: {
            query: searchQuery,
            page: pagination.page,
            page_size: pagination.page_size
          }
        });
        data = response.data;
      } else {
        const response = await listDatasourceTablesApiV1DatasourcesDatasourceNameTablesGet({
          path: { datasource_name: datasourceName },
          query: {
            page: pagination.page,
            page_size: pagination.page_size
          }
        });
        data = response.data;
      }

      if (data) {
        setTables(data.tables);
        setPagination(data.pagination);
      }
    } catch (error) {
      console.error('Failed to load tables:', error);
      setTables([]);
      setPagination({
        page: 1,
        page_size: 20,
        total_count: 0,
        has_next: false,
        has_previous: false
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setIsSearchMode(false);
      setPagination(prev => ({ ...prev, page: 1 }));
      loadTables();
      return;
    }
    
    setIsSearchMode(true);
    setPagination(prev => ({ ...prev, page: 1 }));
    loadTables();
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    setIsSearchMode(false);
    setPagination(prev => ({ ...prev, page: 1 }));
    loadTables();
  };

  const handlePageChange = (newPage: number) => {
    setPagination(prev => ({ ...prev, page: newPage }));
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat().format(num);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center">
              <Table className="w-5 h-5 mr-2" />
              {datasourceInfo.display_name}
              <Badge className="ml-2" variant="outline">
                {formatNumber(pagination.total_count)} tables
              </Badge>
            </div>
            <Button onClick={loadTables} variant="outline" size="sm">
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Search */}
          {datasourceInfo.capabilities.includes('search') && (
            <div className="flex space-x-2">
              <div className="flex-1">
                <Input
                  placeholder="Search tables..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                />
              </div>
              <Button onClick={handleSearch} disabled={loading}>
                <Search className="w-4 h-4 mr-2" />
                Search
              </Button>
              {isSearchMode && (
                <Button onClick={handleClearSearch} variant="outline">
                  Clear
                </Button>
              )}
            </div>
          )}

          {/* Search Results Info */}
          {isSearchMode && (
            <div className="text-sm text-gray-600">
              Found {formatNumber(pagination.total_count)} tables matching "{searchQuery}"
            </div>
          )}

          {/* Tables List */}
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="flex items-center space-x-2">
                <RefreshCw className="h-4 w-4 animate-spin" />
                <span>Loading tables...</span>
              </div>
            </div>
          ) : tables.length === 0 ? (
            <div className="text-center py-8">
              <FileText className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                {isSearchMode ? 'No tables found' : 'No tables available'}
              </h3>
              <p className="text-gray-500">
                {isSearchMode 
                  ? `No tables match your search for "${searchQuery}"`
                  : 'This data source doesn\'t contain any tables yet.'
                }
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {tables.map((tableName) => (
                <div
                  key={tableName}
                  className="flex items-center justify-between p-3 border rounded-lg hover:shadow-sm transition-shadow"
                >
                  <div className="flex items-center space-x-3">
                    <div className="p-1.5 bg-blue-50 rounded">
                      <Table className="h-4 w-4 text-blue-600" />
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900">{tableName}</h4>
                    </div>
                  </div>
                  
                  {datasourceInfo.capabilities.includes('preview') && (
                    <Button
                      onClick={() => setPreviewTable(tableName)}
                      variant="outline"
                      size="sm"
                    >
                      <Eye className="w-4 h-4 mr-2" />
                      Preview
                    </Button>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Pagination */}
          {pagination.total_count > pagination.page_size && (
            <div className="flex items-center justify-between pt-4">
              <div className="text-sm text-gray-600">
                Showing {(pagination.page - 1) * pagination.page_size + 1} to{' '}
                {Math.min(pagination.page * pagination.page_size, pagination.total_count)} of{' '}
                {formatNumber(pagination.total_count)} tables
              </div>
              <div className="flex space-x-2">
                <Button
                  onClick={() => handlePageChange(pagination.page - 1)}
                  disabled={!pagination.has_previous}
                  variant="outline"
                  size="sm"
                >
                  <ChevronLeft className="w-4 h-4" />
                  Previous
                </Button>
                <span className="flex items-center px-3 py-1 text-sm">
                  Page {pagination.page}
                </span>
                <Button
                  onClick={() => handlePageChange(pagination.page + 1)}
                  disabled={!pagination.has_next}
                  variant="outline"
                  size="sm"
                >
                  Next
                  <ChevronRight className="w-4 h-4 ml-1" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Table Preview Dialog */}
      {previewTable && (
        <TablePreviewDialog
          datasourceName={datasourceName}
          tableName={previewTable}
          onClose={() => setPreviewTable(null)}
        />
      )}
    </>
  );
};