import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { AlertDialog, AlertDialogContent, AlertDialogHeader, AlertDialogTitle, AlertDialogDescription, AlertDialogFooter, AlertDialogAction, AlertDialogCancel } from "@/components/ui/alert-dialog";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from "@/components/ui/dropdown-menu";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { MoreVertical, Edit2, Trash2, Grid3X3, List, ChevronLeft, ChevronRight, Upload, Download, Share, Loader2, Network, FileCode, FolderPlus, FolderIcon, Menu, X } from "lucide-react";
import { listProjectsApiV1ProjectsGet, deleteProjectApiV1ProjectsProjectIdDelete, updateProjectApiV1ProjectsProjectIdPut, moveProjectApiV1ProjectsProjectIdMovePost } from "@/api";
import { CreateProjectWizard } from './CreateProjectWizard';
import { ImportProjectDialog } from './ImportProjectDialog';
import { ShareProjectDialog } from './ShareProjectDialog';
import { MoveProjectDialog } from './MoveProjectDialog';
import { useFolders } from '@/contexts/FolderContext';
import {
  FolderTreeView,
  FolderBreadcrumb,
  CreateFolderDialog,
  RenameFolderDialog,
  DeleteFolderDialog,
  MoveFolderDialog,
  FolderShareDialog,
} from '@/components/folders';
import type { Folder } from '@/types/folder-types';

import type { ProjectResponse, ProjectType } from "@/api/types.gen";

// Use the generated type directly
type Project = ProjectResponse;

export const ProjectListPage: React.FC = () => {
  const navigate = useNavigate();
  const {
    currentFolder,
    folderHierarchy,
    folders,
    navigateToFolder,
    navigateToRoot,
    error: folderError,
    loading: folderLoading,
    refreshFolders,
  } = useFolders();

  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newProjectType, setNewProjectType] = useState<ProjectType>('tree');
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [shareDialog, setShareDialog] = useState<{ open: boolean; project: Project | null }>({ open: false, project: null });
  const [renameDialog, setRenameDialog] = useState<{ open: boolean; project: Project | null }>({ open: false, project: null });
  const [deleteDialog, setDeleteDialog] = useState<{ open: boolean; project: Project | null }>({ open: false, project: null });
  const [moveDialog, setMoveDialog] = useState<{ open: boolean; project: Project | null }>({ open: false, project: null });
  const [renameForm, setRenameForm] = useState({ name: '', description: '' });
  const [operationLoading, setOperationLoading] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(12);
  const [totalProjects, setTotalProjects] = useState(0);
  const [totalPages, setTotalPages] = useState(0);

  // Folder dialogs state
  const [createFolderOpen, setCreateFolderOpen] = useState(false);
  const [renameFolderDialog, setRenameFolderDialog] = useState<{ open: boolean; folder: Folder | null }>({ open: false, folder: null });
  const [deleteFolderDialog, setDeleteFolderDialog] = useState<{ open: boolean; folder: Folder | null }>({ open: false, folder: null });
  const [moveFolderDialog, setMoveFolderDialog] = useState<{ open: boolean; folder: Folder | null }>({ open: false, folder: null });
  const [shareFolderDialog, setShareFolderDialog] = useState<{ open: boolean; folder: Folder | null }>({ open: false, folder: null });
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const fetchProjects = useCallback(async (page: number, size: number, search = '', folderId: string | null = null) => {
    try {
      setLoading(true);
      setError(null);

      // Use the generated SDK function with folder filter
      const response = await listProjectsApiV1ProjectsGet({
        query: {
          page,
          size,
          ...(search && { search }),
          ...(folderId !== undefined && { folder_id: folderId }),
        }
      });

      if (response.data) {
        setProjects(response.data.items || []);
        setTotalProjects(response.data.total || 0);
        setTotalPages(response.data.pages || 0);
        setCurrentPage(response.data.page || 1);
      }
    } catch (error) {
      console.error('Failed to fetch projects:', error);
      setError('Failed to load projects');
    } finally {
      setLoading(false);
    }
  }, []);

  // Debounce searchTerm to avoid excessive API calls
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchTerm(searchTerm);
    }, 300);

    return () => clearTimeout(timer);
  }, [searchTerm]);

  // Reset to page 1 when search term or items per page changes
  useEffect(() => {
    setCurrentPage(1);
  }, [debouncedSearchTerm, itemsPerPage]);

  // Fetch projects when page, size, search, or folder changes
  useEffect(() => {
    fetchProjects(currentPage, itemsPerPage, debouncedSearchTerm, currentFolder?.id || null);
  }, [currentPage, itemsPerPage, debouncedSearchTerm, currentFolder]);

  const handleCreateProject = () => {
    setCreateDialogOpen(true);
  };

  const handleProjectCreated = (projectId: string) => {
    fetchProjects(1, itemsPerPage, debouncedSearchTerm); // Refresh the project list
    // Navigate to the correct editor based on project type
    if (newProjectType === 'flat_rule') {
      navigate(`/flat-rules/${projectId}`);
    } else {
      navigate(`/projects/${projectId}`);
    }
  };

  const handleProjectImported = (projectId: string, projectType: 'tree' | 'flat_rule') => {
    fetchProjects(1, itemsPerPage, debouncedSearchTerm); // Refresh the project list
    // Navigate to the correct editor based on project type
    if (projectType === 'flat_rule') {
      navigate(`/flat-rules/${projectId}`);
    } else {
      navigate(`/projects/${projectId}`);
    }
  };

  const handleExportProject = async (project: Project) => {
    try {
      setOperationLoading(true);
      
      // Create a temporary link to trigger download
      const link = document.createElement('a');
      link.href = `/api/v1/projects/${project.id}/export`;
      link.download = `${project.name.replace(/[^a-zA-Z0-9]/g, '_')}_export.json`;
      
      // Trigger the download
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
    } catch (error) {
      console.error('Failed to export project:', error);
      setError('Failed to export project');
    } finally {
      setOperationLoading(false);
    }
  };

  const handleRenameProject = (project: Project) => {
    setRenameForm({ name: project.name, description: project.description || '' });
    setRenameDialog({ open: true, project });
  };

  const handleShareProject = (project: Project) => {
    setShareDialog({ open: true, project });
  };

  const handleDeleteProject = (project: Project) => {
    setDeleteDialog({ open: true, project });
  };

  const handleMoveProject = (project: Project) => {
    setMoveDialog({ open: true, project });
  };

  const submitMoveProject = async (projectId: string, newFolderId: string | null) => {
    try {
      setOperationLoading(true);

      await moveProjectApiV1ProjectsProjectIdMovePost({
        path: {
          project_id: projectId
        },
        body: {
          new_folder_id: newFolderId
        }
      });

      setMoveDialog({ open: false, project: null });
      await fetchProjects(currentPage, itemsPerPage, debouncedSearchTerm, currentFolder?.id ?? null);
    } catch (error) {
      console.error('Failed to move project:', error);
      setError('Failed to move project');
      throw error; // Re-throw to let dialog handle it
    } finally {
      setOperationLoading(false);
    }
  };

  const submitRename = async () => {
    if (!renameDialog.project) return;

    try {
      setOperationLoading(true);

      await updateProjectApiV1ProjectsProjectIdPut({
        path: {
          project_id: renameDialog.project.id
        },
        body: {
          name: renameForm.name,
          description: renameForm.description || undefined
        }
      });

      setRenameDialog({ open: false, project: null });
      fetchProjects(currentPage, itemsPerPage, debouncedSearchTerm); // Refresh the list
    } catch (error) {
      console.error('Failed to rename project:', error);
      setError('Failed to rename project');
    } finally {
      setOperationLoading(false);
    }
  };

  const submitDelete = async () => {
    if (!deleteDialog.project) return;

    try {
      setOperationLoading(true);

      // Call the generated SDK function
      await deleteProjectApiV1ProjectsProjectIdDelete({
        path: {
          project_id: deleteDialog.project.id
        }
      });

      setDeleteDialog({ open: false, project: null });
      fetchProjects(1, itemsPerPage, debouncedSearchTerm); // Refresh from first page after delete
    } catch (error) {
      console.error('Failed to delete project:', error);
      setError('Failed to delete project');
    } finally {
      setOperationLoading(false);
    }
  };

  const handleOpenProject = (project: Project) => {
    // Both tree and flat_rule projects now use the same endpoint
    // The flat-rules route still exists but uses old API - needs refactoring
    // For now, route flat_rule to flat-rules editor, tree to tree editor
    if (project.project_type === 'flat_rule') {
      navigate(`/flat-rules/${project.id}`);
    } else {
      navigate(`/projects/${project.id}`);
    }
  };

  // Folder handlers
  const handleCreateFolder = (parentId?: string) => {
    setCreateFolderOpen(true);
  };

  const handleFolderCreated = () => {
    // Refresh projects in case any were affected
    fetchProjects(currentPage, itemsPerPage, debouncedSearchTerm, currentFolder?.id || null);
  };

  const handleRenameFolder = (folder: Folder) => {
    setRenameFolderDialog({ open: true, folder });
  };

  const handleDeleteFolder = (folder: Folder) => {
    setDeleteFolderDialog({ open: true, folder });
  };

  const handleShareFolder = (folder: Folder) => {
    setShareFolderDialog({ open: true, folder });
  };

  const handleMoveFolder = (folder: Folder) => {
    setMoveFolderDialog({ open: true, folder });
  };

  const handleFolderSelect = async (folderId: string | null) => {
    await navigateToFolder(folderId);
    setCurrentPage(1); // Reset to first page when changing folders
  };

  // Projects are already filtered and paginated by the backend
  const paginatedProjects = projects;
  const startIndex = (currentPage - 1) * itemsPerPage;

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <div
        className={`bg-white border-r transition-all duration-300 ${
          sidebarOpen ? 'w-64' : 'w-0'
        } overflow-hidden flex-shrink-0`}
      >
        <div className="h-full flex flex-col">
          <div className="p-4 border-b flex items-center justify-between">
            <h2 className="font-semibold text-gray-700 flex items-center gap-2">
              <FolderIcon className="h-5 w-5" />
              Folders
            </h2>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="flex-1 overflow-y-auto">
            <FolderTreeView
              folders={folders}
              currentFolderId={currentFolder?.id || null}
              onFolderSelect={handleFolderSelect}
              onCreateFolder={handleCreateFolder}
              onRenameFolder={handleRenameFolder}
              onDeleteFolder={handleDeleteFolder}
              onShareFolder={handleShareFolder}
              onMoveFolder={handleMoveFolder}
              loading={folderLoading}
              error={folderError}
              onRetry={refreshFolders}
            />
          </div>
          <div className="p-3 border-t">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleCreateFolder()}
              className="w-full"
            >
              <FolderPlus className="h-4 w-4 mr-2" />
              New Folder
            </Button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 p-6 overflow-auto">
        <div className="max-w-7xl mx-auto">
          {/* Mobile sidebar toggle */}
          {!sidebarOpen && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSidebarOpen(true)}
              className="mb-4 lg:hidden"
            >
              <Menu className="h-4 w-4 mr-2" />
              Show Folders
            </Button>
          )}

          {/* Breadcrumb */}
          {folderHierarchy.length > 0 && (
            <div className="mb-4">
              <FolderBreadcrumb
                hierarchy={folderHierarchy}
                onNavigate={handleFolderSelect}
              />
            </div>
          )}

          <div className="flex justify-between items-center mb-6">
            <h1 className="text-3xl font-bold">
              {currentFolder ? currentFolder.name : 'All Projects'}
            </h1>
            <div className="flex space-x-2">
              <Button variant="outline" onClick={() => setImportDialogOpen(true)}>
                <Upload className="w-4 h-4 mr-2" />
                Import Project
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button>
                    Create New Project
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => {
                    setCreateDialogOpen(true);
                    setNewProjectType('tree');
                  }}>
                    <Network className="mr-2 h-4 w-4 text-blue-600" />
                    Create Tree Project
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => {
                    setCreateDialogOpen(true);
                    setNewProjectType('flat_rule');
                  }}>
                    <FileCode className="mr-2 h-4 w-4 text-green-600" />
                    Create Flat Rule Project
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>

        <div className="flex justify-between items-center mb-6">
          <Input
            placeholder="Search projects..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="max-w-md"
          />
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">View:</span>
              <div className="flex border rounded-md">
                <Button
                  variant={viewMode === 'grid' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('grid')}
                  className="rounded-r-none"
                >
                  <Grid3X3 className="h-4 w-4" />
                </Button>
                <Button
                  variant={viewMode === 'list' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('list')}
                  className="rounded-l-none"
                >
                  <List className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Show:</span>
              <Select value={itemsPerPage.toString()} onValueChange={(value) => setItemsPerPage(Number(value))}>
                <SelectTrigger className="w-20">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="6">6</SelectItem>
                  <SelectItem value="12">12</SelectItem>
                  <SelectItem value="24">24</SelectItem>
                  <SelectItem value="48">48</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {error ? (
          <Card>
            <CardContent className="pt-12 pb-12 text-center">
              <div className="text-red-600 mb-4">{error}</div>
              <Button onClick={() => fetchProjects(currentPage, itemsPerPage, debouncedSearchTerm, currentFolder?.id || null)}>
                Retry
              </Button>
            </CardContent>
          </Card>
        ) : loading ? (
          <Card>
            <CardContent className="pt-12 pb-12 text-center">
              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-gray-400" />
              <div className="text-gray-500">Loading projects...</div>
            </CardContent>
          </Card>
        ) : totalProjects === 0 ? (
          <Card>
            <CardContent className="pt-6 text-center">
              <div className="text-gray-500">
                {searchTerm ? 'No projects match your search' : 'No projects found'}
              </div>
              {!searchTerm && (
                <Button onClick={handleCreateProject} className="mt-4">
                  Create Your First Project
                </Button>
              )}
            </CardContent>
          </Card>
        ) : (
          <>
            {viewMode === 'grid' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {paginatedProjects.map((project) => (
                  <Card key={project.id} data-testid="project-card" className="hover:shadow-lg transition-shadow">
                    <CardHeader>
                      <div className="flex justify-between items-start">
                        <div className="flex items-center gap-2">
                          {project.project_type === 'tree' ? (
                            <Network className="h-5 w-5 text-blue-600 flex-shrink-0" />
                          ) : (
                            <FileCode className="h-5 w-5 text-green-600 flex-shrink-0" />
                          )}
                          <CardTitle className="text-lg">{project.name}</CardTitle>
                        </div>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="sm" className="h-8 w-8 p-0 flex-shrink-0" data-testid="project-list-menu">
                              <MoreVertical className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => handleShareProject(project)}>
                              <Share className="mr-2 h-4 w-4" />
                              Share
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleExportProject(project)}>
                              <Download className="mr-2 h-4 w-4" />
                              Export
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem onClick={() => handleRenameProject(project)}>
                              <Edit2 className="mr-2 h-4 w-4" />
                              Rename
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleMoveProject(project)}>
                              <FolderIcon className="mr-2 h-4 w-4" />
                              Move to Folder
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem
                              onClick={() => handleDeleteProject(project)}
                              className="text-red-600 focus:text-red-600"
                            >
                              <Trash2 className="mr-2 h-4 w-4" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                      <CardDescription className="line-clamp-2">
                        {project.description || 'No description available'}
                      </CardDescription>
                      <div className="flex items-center justify-between text-sm text-muted-foreground pt-2">
                        <span>Updated: {new Date(project.updated_at).toLocaleDateString()}</span>
                        {/* TODO: Add folder badge when project.folder_id is available
                        {project.folder_id && (
                          <span className="flex items-center gap-1 text-xs bg-gray-100 px-2 py-1 rounded">
                            <FolderIcon className="h-3 w-3" />
                            {folders.find(f => f.id === project.folder_id)?.name}
                          </span>
                        )}
                        */}
                      </div>
                    </CardHeader>
                    <CardContent>
                      <Button 
                        onClick={() => handleOpenProject(project)}
                        className="w-full"
                      >
                        Open Project
                      </Button>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="border rounded-lg">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Description</TableHead>
                      <TableHead>Last Updated</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {paginatedProjects.map((project) => (
                      <TableRow key={project.id} data-testid="project-row" className="hover:bg-muted/50">
                        <TableCell className="font-medium">
                          <button
                            onClick={() => handleOpenProject(project)}
                            className="text-left hover:underline text-blue-600 hover:text-blue-800 flex items-center gap-2"
                          >
                            {project.project_type === 'tree' ? (
                              <Network className="h-4 w-4 text-blue-600 flex-shrink-0" />
                            ) : (
                              <FileCode className="h-4 w-4 text-green-600 flex-shrink-0" />
                            )}
                            {project.name}
                          </button>
                        </TableCell>
                        <TableCell className="text-muted-foreground max-w-xs">
                          <div className="truncate">
                            {project.description || 'No description available'}
                          </div>
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {new Date(project.updated_at).toLocaleDateString()}
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-2">
                            <Button 
                              onClick={() => handleOpenProject(project)}
                              variant="outline"
                              size="sm"
                            >
                              Open
                            </Button>
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="sm" className="h-8 w-8 p-0" data-testid="project-list-menu">
                                  <MoreVertical className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuItem onClick={() => handleShareProject(project)}>
                                  <Share className="mr-2 h-4 w-4" />
                                  Share
                                </DropdownMenuItem>
                                <DropdownMenuItem onClick={() => handleExportProject(project)}>
                                  <Download className="mr-2 h-4 w-4" />
                                  Export
                                </DropdownMenuItem>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem onClick={() => handleRenameProject(project)}>
                                  <Edit2 className="mr-2 h-4 w-4" />
                                  Rename
                                </DropdownMenuItem>
                                <DropdownMenuItem onClick={() => handleMoveProject(project)}>
                                  <FolderIcon className="mr-2 h-4 w-4" />
                                  Move to Folder
                                </DropdownMenuItem>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem
                                  onClick={() => handleDeleteProject(project)}
                                  className="text-red-600 focus:text-red-600"
                                >
                                  <Trash2 className="mr-2 h-4 w-4" />
                                  Delete
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
            
            {/* Pagination */}
            {!loading && totalPages > 1 && (
              <div className="flex items-center justify-between mt-8">
                <div className="text-sm text-gray-600">
                  Showing {startIndex + 1}-{Math.min(startIndex + itemsPerPage, totalProjects)} of {totalProjects} projects
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                    disabled={currentPage === 1}
                  >
                    <ChevronLeft className="h-4 w-4" />
                    Previous
                  </Button>
                  <div className="flex items-center gap-1">
                    {Array.from({ length: totalPages }, (_, i) => i + 1)
                      .filter(page => {
                        return page === 1 || page === totalPages || Math.abs(page - currentPage) <= 1;
                      })
                      .map((page, index, filteredPages) => {
                        const showEllipsis = index > 0 && filteredPages[index - 1] !== page - 1;
                        return (
                          <React.Fragment key={page}>
                            {showEllipsis && <span className="px-2 text-gray-400">...</span>}
                            <Button
                              variant={currentPage === page ? "default" : "outline"}
                              size="sm"
                              onClick={() => setCurrentPage(page)}
                              className="w-8 h-8 p-0"
                            >
                              {page}
                            </Button>
                          </React.Fragment>
                        );
                      })}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                    disabled={currentPage === totalPages}
                  >
                    Next
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
        
        <CreateProjectWizard
          open={createDialogOpen}
          onOpenChange={setCreateDialogOpen}
          onProjectCreated={handleProjectCreated}
          projectType={newProjectType}
        />
        
        <ImportProjectDialog
          open={importDialogOpen}
          onOpenChange={setImportDialogOpen}
          onProjectImported={handleProjectImported}
        />
        
        <ShareProjectDialog
          open={shareDialog.open}
          onOpenChange={(open) => setShareDialog({ open, project: shareDialog.project })}
          project={shareDialog.project}
          onShareAdded={() => {
            // Refresh project list to potentially update shared indicators
            fetchProjects(currentPage, itemsPerPage, debouncedSearchTerm);
          }}
        />

        <MoveProjectDialog
          open={moveDialog.open}
          project={moveDialog.project}
          folders={folders}
          currentFolderId={moveDialog.project?.folder_id ?? null}
          onClose={() => setMoveDialog({ open: false, project: null })}
          onConfirm={submitMoveProject}
        />

        {/* Rename Dialog */}
        <Dialog open={renameDialog.open} onOpenChange={(open) => setRenameDialog({ open, project: renameDialog.project })}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Rename Project</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Project Name</label>
                <Input
                  value={renameForm.name}
                  onChange={(e) => setRenameForm({ ...renameForm, name: e.target.value })}
                  placeholder="Enter project name"
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Description</label>
                <Input
                  value={renameForm.description}
                  onChange={(e) => setRenameForm({ ...renameForm, description: e.target.value })}
                  placeholder="Enter project description (optional)"
                  className="mt-1"
                />
              </div>
              <div className="flex justify-end space-x-2 pt-4">
                <Button 
                  variant="outline" 
                  onClick={() => setRenameDialog({ open: false, project: null })}
                  disabled={operationLoading}
                >
                  Cancel
                </Button>
                <Button 
                  onClick={submitRename}
                  disabled={operationLoading || !renameForm.name.trim()}
                >
                  {operationLoading ? 'Saving...' : 'Save Changes'}
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
        
        {/* Delete Confirmation Dialog */}
        <AlertDialog open={deleteDialog.open} onOpenChange={(open) => setDeleteDialog({ open, project: deleteDialog.project })}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete Project</AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to delete "{deleteDialog.project?.name}"? This action cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel disabled={operationLoading}>
                Cancel
              </AlertDialogCancel>
              <AlertDialogAction
                onClick={submitDelete}
                disabled={operationLoading}
                className="bg-red-600 hover:bg-red-700"
              >
                {operationLoading ? 'Deleting...' : 'Delete Project'}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        {/* Folder Dialogs */}
        <CreateFolderDialog
          open={createFolderOpen}
          onOpenChange={setCreateFolderOpen}
          onFolderCreated={handleFolderCreated}
          defaultParentId={currentFolder?.id}
        />

        <RenameFolderDialog
          open={renameFolderDialog.open}
          onOpenChange={(open) => setRenameFolderDialog({ open, folder: renameFolderDialog.folder })}
          folder={renameFolderDialog.folder}
          onFolderRenamed={handleFolderCreated}
        />

        <DeleteFolderDialog
          open={deleteFolderDialog.open}
          onOpenChange={(open) => setDeleteFolderDialog({ open, folder: deleteFolderDialog.folder })}
          folder={deleteFolderDialog.folder}
          onFolderDeleted={handleFolderCreated}
        />

        <MoveFolderDialog
          open={moveFolderDialog.open}
          onOpenChange={(open) => setMoveFolderDialog({ open, folder: moveFolderDialog.folder })}
          folder={moveFolderDialog.folder}
          onFolderMoved={handleFolderCreated}
        />

        <FolderShareDialog
          open={shareFolderDialog.open}
          onOpenChange={(open) => setShareFolderDialog({ open, folder: shareFolderDialog.folder })}
          folder={shareFolderDialog.folder}
          onShareAdded={handleFolderCreated}
        />
        </div>
      </div>
    </div>
  );
};