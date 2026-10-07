import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Database,
  Upload,
  Search,
  Eye,
  RefreshCw,
  CloudUpload,
  Table,
  HardDrive,
  Code
} from 'lucide-react';
import { DataSourceExplorer } from './DataSourceExplorer';
import { FileUploader } from './FileUploader';
import { CreateViewDialog } from './CreateViewDialog';
import { DataSourceInfo, DataSourceList } from '../../types/datasource-types';
import { listDatasourcesApiV1DatasourcesGet } from "@/api";

export const DatasetsPage: React.FC = () => {
  const [datasources, setDatasources] = useState<Record<string, DataSourceInfo>>({});
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<string>('discover');
  const [showCreateViewDialog, setShowCreateViewDialog] = useState(false);

  useEffect(() => {
    fetchDatasources();
  }, []);

  const fetchDatasources = async () => {
    try {
      setLoading(true);
      const response = await listDatasourcesApiV1DatasourcesGet();

      if (response.data) {
        // Convert array to Record for easier access in UI
        const datasourceRecord: Record<string, DataSourceInfo> = {};
        response.data.datasources.forEach(ds => {
          datasourceRecord[ds.name] = {
            ...ds,
            description: ds.description ?? undefined, // Convert null to undefined for consistency
            icon: ds.icon ?? undefined, // Convert null to undefined for consistency
          };
        });
        setDatasources(datasourceRecord);

        // If we have upload-capable sources, set the default tab appropriately
        const hasUploadSources = Object.values(datasourceRecord).some(ds => ds.capabilities.includes('upload'));
        if (!hasUploadSources && Object.keys(datasourceRecord).length > 0) {
          setActiveTab('discover');
        }
      }
    } catch (error) {
      console.error('Failed to fetch datasources:', error);
      setDatasources({});
    } finally {
      setLoading(false);
    }
  };

  const getProviderIcon = (provider: string) => {
    switch (provider) {
      case 's3':
        return Database;
      case 'upload':
        return CloudUpload;
      case 'postgres':
        return Database;
      default:
        return HardDrive;
    }
  };

  const getProviderBadgeColor = (provider: string) => {
    switch (provider) {
      case 's3':
        return 'bg-orange-100 text-orange-800';
      case 'upload':
        return 'bg-blue-100 text-blue-800';
      case 'postgres':
        return 'bg-green-100 text-green-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const discoverableDataSources = Object.entries(datasources).filter(([_, ds]) => ds.capabilities.includes('discovery'));
  const uploadableDataSources = Object.entries(datasources).filter(([_, ds]) => ds.capabilities.includes('upload'));
  // For creating views: need discoverable sources (to read from) AND uploadable sources (to save to)
  const canCreateViews = discoverableDataSources.length > 0 && uploadableDataSources.length > 0;

  const handleViewCreated = (viewName: string, datasourceName: string) => {
    // Refresh datasources to show the new view
    fetchDatasources();
    setShowCreateViewDialog(false);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="flex items-center space-x-2">
          <RefreshCw className="h-4 w-4 animate-spin" />
          <span>Loading datasets...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dataset Management</h1>
          <p className="text-gray-600 mt-2">
            Discover, upload, and manage datasets across your configured data sources.
          </p>
        </div>
        <Button onClick={fetchDatasources} variant="outline">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {Object.keys(datasources).length === 0 ? (
        <Card>
          <CardContent className="text-center py-12">
            <Database className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Data Sources Configured</h3>
            <p className="text-gray-500 mb-4">
              Contact your administrator to configure data sources for dataset management.
            </p>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Data Sources Overview */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Database className="w-5 h-5 mr-2" />
                Available Data Sources
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(datasources).map(([name, datasource]) => {
                  const IconComponent = getProviderIcon(datasource.provider);
                  
                  return (
                    <Card key={name} className="border">
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center space-x-3">
                            <div className="p-2 bg-blue-50 rounded-lg">
                              <IconComponent className="h-5 w-5 text-blue-600" />
                            </div>
                            <div>
                              <h4 className="font-medium text-gray-900">{datasource.display_name}</h4>
                              <Badge className={getProviderBadgeColor(datasource.provider)}>
                                {datasource.provider}
                              </Badge>
                            </div>
                          </div>
                        </div>

                        {datasource.description && (
                          <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                            {datasource.description}
                          </p>
                        )}

                        <div className="flex flex-wrap gap-1">
                          {datasource.capabilities.includes('discovery') && (
                            <Badge variant="outline" className="text-xs">
                              <Search className="w-3 h-3 mr-1" />
                              Discovery
                            </Badge>
                          )}
                          {datasource.capabilities.includes('upload') && (
                            <Badge variant="outline" className="text-xs">
                              <Upload className="w-3 h-3 mr-1" />
                              Upload
                            </Badge>
                          )}
                          {datasource.capabilities.includes('preview') && (
                            <Badge variant="outline" className="text-xs">
                              <Eye className="w-3 h-3 mr-1" />
                              Preview
                            </Badge>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {/* Main Tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="discover" disabled={discoverableDataSources.length === 0}>
                <Search className="w-4 h-4 mr-2" />
                Discover Data ({discoverableDataSources.length})
              </TabsTrigger>
              <TabsTrigger value="upload" disabled={uploadableDataSources.length === 0}>
                <CloudUpload className="w-4 h-4 mr-2" />
                Upload Data ({uploadableDataSources.length})
              </TabsTrigger>
              <TabsTrigger value="create-view" disabled={!canCreateViews}>
                <Code className="w-4 h-4 mr-2" />
                Create View
              </TabsTrigger>
            </TabsList>

            <TabsContent value="discover" className="space-y-6">
              {discoverableDataSources.length === 0 ? (
                <Card>
                  <CardContent className="text-center py-8">
                    <Search className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Discoverable Data Sources</h3>
                    <p className="text-gray-500">
                      No data sources are configured for data discovery.
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <div className="space-y-6">
                  {discoverableDataSources.map(([name, datasource]) => (
                    <DataSourceExplorer
                      key={name}
                      datasourceName={name}
                      datasourceInfo={datasource}
                    />
                  ))}
                </div>
              )}
            </TabsContent>

            <TabsContent value="upload" className="space-y-6">
              {uploadableDataSources.length === 0 ? (
                <Card>
                  <CardContent className="text-center py-8">
                    <CloudUpload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Upload-Enabled Data Sources</h3>
                    <p className="text-gray-500">
                      No data sources are configured for file uploads.
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <div className="space-y-6">
                  {uploadableDataSources.map(([name, datasource]) => (
                    <FileUploader
                      key={name}
                      datasourceName={name}
                      datasourceInfo={datasource}
                    />
                  ))}
                </div>
              )}
            </TabsContent>

            <TabsContent value="create-view" className="space-y-6">
              {!canCreateViews ? (
                <Card>
                  <CardContent className="text-center py-8">
                    <Code className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">Cannot Create Views</h3>
                    <p className="text-gray-500">
                      Creating views requires at least one datasource with discovery capability (to read from)
                      and one datasource with upload capability (to save to).
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <Code className="w-5 h-5 mr-2" />
                      Create View from SQL Query
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <p className="text-sm text-gray-600">
                        Execute SQL queries against existing tables to create new materialized views.
                        Views are saved as parquet files and can be used like any other dataset.
                      </p>
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <h4 className="font-medium text-blue-900 mb-2">How it works:</h4>
                        <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
                          <li>Select a source datasource and table to query</li>
                          <li>Select a destination datasource to save the result</li>
                          <li>Write a SQL query to filter, aggregate, or transform the data</li>
                          <li>Preview the results to verify your query</li>
                          <li>Save the output as a new view in the destination datasource</li>
                          <li>Use the view in Features tab like any other table</li>
                        </ol>
                      </div>
                      <Button onClick={() => setShowCreateViewDialog(true)} size="lg" className="w-full">
                        <Code className="w-4 h-4 mr-2" />
                        Create New View
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>
          </Tabs>
        </>
      )}

      {/* Create View Dialog */}
      <CreateViewDialog
        open={showCreateViewDialog}
        onClose={() => setShowCreateViewDialog(false)}
        onViewCreated={handleViewCreated}
      />
    </div>
  );
};