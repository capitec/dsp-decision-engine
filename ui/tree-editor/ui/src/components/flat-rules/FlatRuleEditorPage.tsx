import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Loader2, ArrowLeft, TableIcon, Save, Download, Database, FileInput, Edit2 } from "lucide-react";
import { flatRuleApi } from './api';
import { RuleEditor } from './RuleEditor';
import {
  useFlatRuleSaving,
  useFlatRuleError,
  useFlatRuleSaveProject,
  useFlatRuleInitializeFromProject,
  useFlatRuleHasInputSchemaErrors,
  useFlatRuleHasOutputErrors,
  useFlatRuleHasParameterErrors,
  useFlatRuleRuleErrors
} from './useFlatRuleStore';
import { OutputConfigurationTab } from './OutputConfigurationTab';
import { InputConfigurationTab } from './InputConfigurationTab';
import type { FlatRuleProjectDetail } from './types';
import { useToast } from "@/hooks/use-toast";
import { apiErrorMessage } from "@/lib/utils";
import { parseFlatRuleValidationErrors, formatRuleErrors } from "@/utils/flat-rule-validation-parser";

// Header component that can access store for save functionality
const FlatRuleEditorHeader: React.FC<{
  project: FlatRuleProjectDetail,
  onBack: () => void,
  onExport: () => void,
  onRename: () => void
}> = ({ project, onBack, onExport, onRename }) => {
  const saving = useFlatRuleSaving();
  const error = useFlatRuleError();
  const saveProject = useFlatRuleSaveProject();

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="max-w-6xl mx-auto flex justify-between items-center">
        <div className="flex items-center gap-4">
          <Button onClick={onBack} variant="ghost" size="sm">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>
          <div className="flex items-center gap-3">
            <TableIcon className="w-6 h-6 text-blue-600" />
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold">{project.name}</h1>
                <Button onClick={onRename} variant="ghost" size="sm" className="h-6 w-6 p-0">
                  <Edit2 className="w-3 h-3" />
                </Button>
              </div>
              {project.description && (
                <p className="text-sm text-gray-600">{project.description}</p>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {error && (
            <span className="text-sm text-red-600 mr-2">{error}</span>
          )}
          <Button variant="outline" onClick={onExport}>
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button onClick={saveProject} disabled={saving}>
            {saving ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Save className="w-4 h-4 mr-2" />
            )}
            {saving ? 'Saving...' : 'Save'}
          </Button>
        </div>
      </div>
    </header>
  );
};

export const FlatRuleEditorPage: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [project, setProject] = useState<FlatRuleProjectDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [renameForm, setRenameForm] = useState({ name: '', description: '' });
  const [renameLoading, setRenameLoading] = useState(false);
  const initializeFromProject = useFlatRuleInitializeFromProject();
  const saveProject = useFlatRuleSaveProject();
  const hasInputSchemaErrors = useFlatRuleHasInputSchemaErrors();
  const hasOutputErrors = useFlatRuleHasOutputErrors();
  const hasParameterErrors = useFlatRuleHasParameterErrors();
  const ruleErrors = useFlatRuleRuleErrors();
  const hasRuleErrors = ruleErrors.size > 0;

  useEffect(() => {
    if (!projectId) return;

    // Always reset state before fetching to ensure clean slate
    setProject(null);
    setError(null);

    const fetchProject = async () => {
      try {
        setLoading(true);
        const projectData = await flatRuleApi.getProject(projectId);
        console.log('[FlatRuleEditor] Project loaded:', projectData);

        setProject(projectData);
        // Initialize the store with the project data
        initializeFromProject(projectData);
      } catch (err: any) {
        console.error('Failed to fetch project:', err);
        setError(apiErrorMessage(err, 'Failed to load project'));
      } finally {
        setLoading(false);
      }
    };

    fetchProject();
  }, [projectId, initializeFromProject]); // Depend on projectId and initializeFromProject

  const handleBack = () => {
    navigate('/flat-rules');
  };

  // Development helper: detect if page reloaded but data didn't
  useEffect(() => {
    if (process.env.NODE_ENV === 'development' && !loading && !error && project && project.project_data) {
      const trees = project.project_data.rules || [];
      if (trees.length === 0) {
        console.warn('[FlatRuleEditor] Project loaded but has no rules. This might be expected for new projects.');
      }
    }
  }, [loading, error, project]);

  const handleExport = async () => {
    if (!project) return;
    // The export is built from the stored document, so store what the editor shows first (as the node editor does).
    if (!(await saveProject())) return;

    try {
      await flatRuleApi.downloadExport(project.id, project.name);
    } catch (err: any) {
      console.error('Failed to export project:', err);
      toast({
        title: "Export failed",
        description: err.validation_errors?.length
          ? formatRuleErrors(parseFlatRuleValidationErrors(err.validation_errors))
          : apiErrorMessage(err, 'Failed to export project'),
        variant: "destructive",
      });
    }
  };

  const handleRename = () => {
    if (!project) return;
    setRenameForm({ name: project.name, description: project.description || '' });
    setRenameDialogOpen(true);
  };

  const handleRenameSubmit = async () => {
    if (!project) return;

    try {
      setRenameLoading(true);
      await flatRuleApi.updateProject(project.id, {
        name: renameForm.name,
        description: renameForm.description
      });

      // Update local project state
      setProject({ ...project, name: renameForm.name, description: renameForm.description });
      setRenameDialogOpen(false);

      toast({
        title: "Project Updated",
        description: "Project name and description updated successfully",
      });
    } catch (err: any) {
      console.error('Failed to update project:', err);
      toast({
        title: "Error",
        description: err.message || 'Failed to update project',
        variant: "destructive"
      });
    } finally {
      setRenameLoading(false);
    }
  };


  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-6xl mx-auto">
          <Card>
            <CardContent className="pt-12 pb-12 text-center">
              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-gray-400" />
              <div className="text-gray-500">Loading flat rule project...</div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-6xl mx-auto">
          <Card>
            <CardContent className="pt-6">
              <div className="text-center text-red-600 mb-4">
                {error || 'Project not found'}
              </div>

              <div className="flex justify-center gap-2">
                <Button onClick={handleBack} variant="outline">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Projects
                </Button>

              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <FlatRuleEditorHeader
        project={project}
        onBack={handleBack}
        onExport={handleExport}
        onRename={handleRename}
      />

      {/* Main Content with Tabs */}
      <div className="max-w-full mx-auto p-6">
        <Tabs defaultValue="rules" className="w-full">
          <TabsList className="grid w-full grid-cols-3 max-w-2xl">
            <TabsTrigger value="rules" className="flex items-center gap-2 relative">
              <TableIcon className="h-4 w-4" />
              Rules
              {hasRuleErrors && (
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full flex items-center justify-center text-white text-[10px] font-bold">
                  !
                </div>
              )}
            </TabsTrigger>
            <TabsTrigger value="inputs" className="flex items-center gap-2 relative">
              <FileInput className="h-4 w-4" />
              Configure Inputs
              {hasInputSchemaErrors && (
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full flex items-center justify-center text-white text-[10px] font-bold">
                  !
                </div>
              )}
            </TabsTrigger>
            <TabsTrigger value="outputs" className="flex items-center gap-2 relative">
              <Database className="h-4 w-4" />
              Configure Outputs
              {hasOutputErrors && (
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full flex items-center justify-center text-white text-[10px] font-bold">
                  !
                </div>
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="rules" className="mt-6">
            <RuleEditor />
          </TabsContent>

          <TabsContent value="inputs" className="mt-6">
            <InputConfigurationTab />
          </TabsContent>

          <TabsContent value="outputs" className="mt-6">
            <OutputConfigurationTab />
          </TabsContent>
        </Tabs>
      </div>

      {/* Rename Dialog */}
      <Dialog open={renameDialogOpen} onOpenChange={setRenameDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Project Details</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <Label htmlFor="rename-name">Project Name</Label>
              <Input
                id="rename-name"
                value={renameForm.name}
                onChange={(e) => setRenameForm({ ...renameForm, name: e.target.value })}
                placeholder="Enter project name"
                className="mt-1"
              />
            </div>
            <div>
              <Label htmlFor="rename-description">Description</Label>
              <Textarea
                id="rename-description"
                value={renameForm.description}
                onChange={(e) => setRenameForm({ ...renameForm, description: e.target.value })}
                placeholder="Enter project description (optional)"
                className="mt-1"
                rows={4}
              />
            </div>
          </div>
          <div className="flex justify-end gap-2">
            <Button
              variant="outline"
              onClick={() => setRenameDialogOpen(false)}
              disabled={renameLoading}
            >
              Cancel
            </Button>
            <Button
              onClick={handleRenameSubmit}
              disabled={renameLoading || !renameForm.name.trim()}
            >
              {renameLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                'Save Changes'
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};