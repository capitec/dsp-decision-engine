import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { FolderIcon, Database, Plus, Clock, TrendingUp, Network, FileCode } from 'lucide-react';
import { getOverviewApiV1OverviewGet } from "@/api";
import type { OverviewResponse, DataSourceInfo } from "@/api/types.gen";

export const OverviewPage: React.FC = () => {
  const navigate = useNavigate();
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchOverviewData();
  }, []);

  const fetchOverviewData = async () => {
    try {
      setLoading(true);

      const response = await getOverviewApiV1OverviewGet();

      if (response.data) {
        setData(response.data);
      }
    } catch (error) {
      console.error('Failed to fetch overview data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenProject = (projectId: string, projectType: string) => {
    // Route based on project type
    if (projectType === 'flat_rule') {
      navigate(`/flat-rules/${projectId}`);
    } else {
      navigate(`/projects/${projectId}`);
    }
  };

  const getAccessBadgeColor = (access: string) => {
    switch (access) {
      case 'admin': return 'bg-red-100 text-red-800';
      case 'write': return 'bg-yellow-100 text-yellow-800';
      case 'read': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div>Loading overview...</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center py-12">
        <div>Failed to load overview data</div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Welcome Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Overview</h1>
        <p className="text-gray-600 mt-2">
          Welcome back! Here's what's happening with your projects and data sources.
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <FolderIcon className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Projects</p>
                <p className="text-2xl font-bold text-gray-900">{data.total_projects}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg">
                <Database className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Data Sources</p>
                <p className="text-2xl font-bold text-gray-900">{data.total_datasources}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <TrendingUp className="h-6 w-6 text-yellow-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Recent Activity</p>
                <p className="text-2xl font-bold text-gray-900">{data.recent_activity_count}</p>
                <p className="text-xs text-gray-500 mt-1">Last 7 days</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Recent Projects */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="flex items-center">
              <Clock className="w-5 h-5 mr-2" />
              Recent Projects
            </CardTitle>
            <Button variant="outline" size="sm" onClick={() => navigate('/projects')}>
              View all
            </Button>
          </CardHeader>
          <CardContent>
            {data.recent_projects.length === 0 ? (
              <div className="text-center py-8">
                <FolderIcon className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No projects yet</h3>
                <p className="text-gray-500 mb-4">
                  Create your first project to get started.
                </p>
                <Button onClick={() => navigate('/projects')}>
                  <Plus className="w-4 h-4 mr-2" />
                  Create Project
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                {data.recent_projects.map((project) => (
                  <div
                    key={project.id}
                    className="p-4 border rounded-lg hover:shadow-sm transition-shadow cursor-pointer"
                    onClick={() => handleOpenProject(project.id, project.project_type)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0 pr-4">
                        <div className="flex items-center gap-2">
                          {project.project_type === 'tree' ? (
                            <Network className="h-4 w-4 text-blue-600 flex-shrink-0" />
                          ) : (
                            <FileCode className="h-4 w-4 text-green-600 flex-shrink-0" />
                          )}
                          <h4 className="font-medium text-gray-900 truncate">{project.name}</h4>
                        </div>
                        {project.description && (
                          <p className="text-sm text-gray-600 mt-1 line-clamp-2 break-words">
                            {project.description}
                          </p>
                        )}
                        <div className="flex items-center space-x-2 mt-2">
                          <Badge className={getAccessBadgeColor(project.access_mode)}>
                            {project.access_mode}
                          </Badge>
                          {project.is_owner && (
                            <Badge variant="outline">Owner</Badge>
                          )}
                        </div>
                      </div>
                      <div className="text-xs text-gray-500 flex-shrink-0">
                        {new Date(project.updated_at).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Data Sources */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="flex items-center">
              <Database className="w-5 h-5 mr-2" />
              Available Data Sources
            </CardTitle>
          </CardHeader>
          <CardContent>
            {data.datasources.length === 0 ? (
              <div className="text-center py-8">
                <Database className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No data sources configured</h3>
                <p className="text-gray-500">
                  Contact your administrator to configure data sources.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {data.datasources.slice(0, 3).map((datasource: DataSourceInfo) => (
                  <div
                    key={datasource.name}
                    className="p-4 border rounded-lg hover:shadow-sm transition-shadow"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="p-2 bg-blue-100 rounded-lg">
                          <Database className="h-4 w-4 text-blue-600" />
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900">{datasource.display_name}</h4>
                          {datasource.description && (
                            <p className="text-sm text-gray-600 mt-1">{datasource.description}</p>
                          )}
                          <Badge variant="outline" className="mt-1">
                            {datasource.provider}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
                {data.datasources.length > 3 && (
                  <div className="text-center text-sm text-gray-500">
                    +{data.datasources.length - 3} more data sources
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
