import { useCallback, useState } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import { apiErrorMessage } from '@/lib/utils';
import { parseValidationErrors, formatValidationErrors } from '@/utils/validation-error-parser';
import Editor from '../editor/Editor';
import { EditorProvider, useEditorContext } from '../editor/EditorContext';
import { useEditorStore, useEditorFormatVersion } from '../editor/useEditorStore';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { MoreVertical, Trash2, Edit3, Save, FileText, Download } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { toast } from "@/hooks/use-toast";
import { useUnsavedChanges } from "@/hooks/use-unsaved-changes";
import {
  deleteProjectApiV1ProjectsProjectIdDelete,
  updateProjectApiV1ProjectsProjectIdPut,
  exportProjectApiV1ProjectsProjectIdExportGet
} from "@/api";
// Validation now happens automatically on save - no separate validation needed


interface ProjectEditorPageProps {
  projectId: string;
  onGoBack: () => void;
}

/**
 * Inner component that uses EditorContext for all state management
 */
const ProjectEditorContent: React.FC<{ onGoBack: () => void }> = ({ onGoBack }) => {
  const {
    loading,
    saving,
    error,
    isDirty,
    projectName,
    projectDescription,
    setProjectName,
    setProjectDescription,
    saveProject,
    projectId,
  } = useEditorContext();
  const setNodes = useEditorStore((state) => state.setNodes);
  const setEdges = useEditorStore((state) => state.setEdges);
  const setInputSchema = useEditorStore((state) => state.setInputSchema);
  const setOutput = useEditorStore((state) => state.setOutput);
  const setSubtrees = useEditorStore((state) => state.setSubtrees);
  const formatVersion = useEditorFormatVersion();

  // Handle unsaved changes warnings
  const { navigateWithConfirmation } = useUnsavedChanges(
    isDirty,
    "You have unsaved changes in your decision tree. Are you sure you want to leave?"
  );

  // Safe go back handler that checks for unsaved changes
  const handleGoBack = useCallback(() => {
    if (isDirty) {
      if (window.confirm("You have unsaved changes in your decision tree. Are you sure you want to leave?")) {
        onGoBack();
      }
    } else {
      onGoBack();
    }
  }, [isDirty, onGoBack]);

  const [showClearDialog, setShowClearDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  // Validation removed - now happens automatically on save

  const handleClearWorkspace = useCallback(() => {
    setShowClearDialog(true);
  }, []);

  const confirmClearWorkspace = useCallback(() => {
    // Clear all workspace state
    setNodes([]);
    setEdges([]);
    setInputSchema([]);
    setOutput({
      data: [
        { action: 'Allow', description: 'Default Action' },
        { action: 'Alert', description: 'A rule suggests that there is an alert' },
        { action: 'Block', description: 'This is more serious and must be blocked' },
      ],
      default: { action: 'Allow', description: 'Default Action' },
      dtypes: [
        ['action', 'String'],
        ['description', 'String'],
      ],
      type_defs: {},
    });
    setSubtrees([]);
    setShowClearDialog(false);
  }, [setNodes, setEdges, setInputSchema, setOutput, setSubtrees]);

  const handleEditProject = useCallback(() => {
    setEditName(projectName || '');
    setEditDescription(projectDescription || '');
    setShowEditDialog(true);
  }, [projectName, projectDescription]);

  const handleDeleteProject = useCallback(() => {
    setShowDeleteDialog(true);
  }, []);

  const confirmDeleteProject = useCallback(async () => {
    try {
      setIsDeleting(true);

      await deleteProjectApiV1ProjectsProjectIdDelete({
        path: { project_id: projectId as string }
      });

      toast({
        title: "Project deleted",
        description: "The project has been permanently deleted.",
      });

      onGoBack();
    } catch (error: any) {
      console.error('Failed to delete project:', error);
      toast({
        title: "Deletion failed",
        description: error.message || "Failed to delete project. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsDeleting(false);
      setShowDeleteDialog(false);
    }
  }, [projectId, onGoBack]);

  const confirmUpdateProject = useCallback(async () => {
    try {
      setIsUpdating(true);

      await updateProjectApiV1ProjectsProjectIdPut({
        path: { project_id: projectId as string },
        body: {
          name: editName.trim(),
          description: editDescription.trim() || undefined,
        }
      });

      toast({
        title: "Project updated",
        description: "Project details have been saved successfully.",
      });

      // Update the EditorContext state to reflect changes in the UI
      setProjectName(editName.trim());
      setProjectDescription(editDescription.trim() || null);

      setShowEditDialog(false);
    } catch (error: any) {
      console.error('Failed to update project:', error);
      toast({
        title: "Update failed",
        description: error.message || "Failed to update project. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsUpdating(false);
    }
  }, [projectId, editName, editDescription, setProjectName, setProjectDescription]);

  const handleExportProject = useCallback(async () => {
    try {
      setIsExporting(true);
      
      // Force save the project first if there are unsaved changes
      if (isDirty) {
        toast({
          title: "Saving project",
          description: "Saving unsaved changes before export...",
        });
        await saveProject();
      }
      
      // Try to export the project - this might fail with validation errors
      const response = await exportProjectApiV1ProjectsProjectIdExportGet({
        path: { project_id: projectId as string }
      });

      // If we get here, the validation passed and we have the tree config
      // Convert to blob for download
      const jsonString = JSON.stringify(response.data, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      
      // Create a temporary link to trigger download
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `${(projectName || 'project').replace(/[^a-zA-Z0-9]/g, '_')}_export.json`;
      
      // Trigger the download
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up the object URL
      URL.revokeObjectURL(link.href);
      
      toast({
        title: "Export successful",
        description: "Project has been exported successfully.",
      });
      
    } catch (error: any) {
      console.error('Failed to export project:', error);

      // Show error toast
      toast({
        title: "Export failed",
        description: error.validation_errors?.length
          ? formatValidationErrors(parseValidationErrors(error.validation_errors))
          : apiErrorMessage(error, "Failed to export project. Please try again."),
        variant: "destructive",
      });
    } finally {
      setIsExporting(false);
    }
  }, [projectId, projectName, isDirty, saveProject]);

  // Validation removed - now happens automatically on save

  // Show loading state
  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading project...</p>
        </div>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-center max-w-lg">
          <p className="text-red-600 mb-4">Error: {error}</p>

          <div className="flex gap-2 justify-center">
            <button
              onClick={handleGoBack}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
            >
              ← Go Back
            </button>

          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col">
      {/* Fixed Header */}
      <header className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 border-b bg-white shadow-sm backdrop-blur-sm bg-white/95">
        <div className="flex items-center space-x-4 min-w-0 flex-1">
          <Button
            onClick={handleGoBack}
            variant="ghost"
            size="sm"
            className="flex-shrink-0"
          >
            ← Back
          </Button>
          <div className="min-w-0 flex-1">
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2">
                <FileText className="h-5 w-5 text-blue-600" />
                <h1 className="text-xl font-semibold truncate text-gray-900">{projectName}</h1>
              </div>
              {/* Format version badge */}
              <Badge
                variant="outline"
                className="text-xs"
                title="Tree format version"
              >
                v{formatVersion}
              </Badge>
              {isDirty && (
                <Badge variant="secondary" className="text-xs bg-amber-100 text-amber-800 border-amber-200">
                  Unsaved changes
                </Badge>
              )}
            </div>
            {projectDescription && (
              <p className="text-sm text-gray-600 truncate mt-1" title={projectDescription}>
                {projectDescription}
              </p>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <Button
            onClick={saveProject}
            disabled={saving || !isDirty}
            className="flex-shrink-0 bg-blue-600 hover:bg-blue-700 text-white"
            size="sm"
          >
            <Save className="h-4 w-4 mr-2" />
            {saving ? 'Saving...' : 'Save'}
          </Button>
          
          {/* Enhanced Kebab Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="flex-shrink-0" data-testid="project-menu">
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem onClick={handleEditProject}>
                <Edit3 className="h-4 w-4 mr-2" />
                Edit Project Details
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleExportProject} disabled={isExporting}>
                <Download className="h-4 w-4 mr-2" />
                {isExporting ? 'Exporting...' : 'Export Project'}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem 
                onClick={handleClearWorkspace}
                className="text-orange-600 focus:text-orange-600"
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Clear Workspace
              </DropdownMenuItem>
              <DropdownMenuItem 
                onClick={handleDeleteProject}
                className="text-red-600 focus:text-red-600"
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Delete Project
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Editor with top padding to account for fixed header */}
      <div className="flex-1 pt-20">
        <Editor />
      </div>

      {/* Clear Workspace Confirmation Dialog */}
      <AlertDialog open={showClearDialog} onOpenChange={setShowClearDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear Workspace</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to clear the entire workspace? This will remove all nodes and edges. 
              This action can be undone using Ctrl+Z.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={confirmClearWorkspace}
              className="bg-orange-600 hover:bg-orange-700"
            >
              Clear Workspace
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Delete Project Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Project</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to permanently delete "{projectName}"? This action cannot be undone.
              All project data including nodes, edges, and configurations will be lost forever.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={confirmDeleteProject}
              disabled={isDeleting}
              className="bg-red-600 hover:bg-red-700"
            >
              {isDeleting ? 'Deleting...' : 'Delete Project'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Edit Project Dialog */}
      <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              <Edit3 className="h-5 w-5" />
              <span>Edit Project Details</span>
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label htmlFor="edit-name" className="text-sm font-medium">
                Project Name *
              </label>
              <Input
                id="edit-name"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="Enter project name"
                disabled={isUpdating}
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="edit-description" className="text-sm font-medium">
                Description
              </label>
              <Textarea
                id="edit-description"
                value={editDescription}
                onChange={(e) => setEditDescription(e.target.value)}
                placeholder="Enter project description (optional)"
                disabled={isUpdating}
                rows={3}
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowEditDialog(false)}
              disabled={isUpdating}
            >
              Cancel
            </Button>
            <Button
              onClick={confirmUpdateProject}
              disabled={isUpdating || !editName.trim()}
            >
              {isUpdating ? 'Saving...' : 'Save Changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

    </div>
  );
};

/**
 * ProjectEditorPage component that provides EditorContext and renders the editor
 */
export default function ProjectEditorPage({ projectId, onGoBack }: ProjectEditorPageProps) {
  return (
    <ReactFlowProvider>
      <EditorProvider projectId={projectId}>
        <ProjectEditorContent onGoBack={onGoBack} />
      </EditorProvider>
    </ReactFlowProvider>
  );
}