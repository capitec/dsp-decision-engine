import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Loader2,
  ChevronRight,
  ChevronLeft,
  Database,
  Network,
  FileCode
} from "lucide-react";
import { Stepper, type Step } from "@/components/ui/stepper";
import { useToast } from "@/hooks/use-toast";
import { LoadFromDatasetDialog, SchemaFieldList, type SchemaField } from '@/components/shared-tree';
import { createProjectApiV1ProjectsPost } from "@/api";
import type { ProjectCreate } from "@/api/types.gen";

type ProjectType = 'tree' | 'flat_rule';

interface CreateProjectWizardProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onProjectCreated: (projectId: string) => void;
  projectType: ProjectType;
}

interface ProjectFormData {
  name: string;
  description: string;
  project_type: ProjectType;
}

const STEPS: Step[] = [
  { id: 'basic', title: 'Basic Info', description: 'Project details' },
  { id: 'inputs', title: 'Configure Inputs', description: 'Optional' },
  { id: 'outputs', title: 'Configure Outputs', description: 'Optional' }
];

export const CreateProjectWizard: React.FC<CreateProjectWizardProps> = ({
  open,
  onOpenChange,
  onProjectCreated,
  projectType
}) => {
  const { toast } = useToast();
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState<ProjectFormData>({
    name: '',
    description: '',
    project_type: projectType
  });

  // Schema configuration state
  const [inputSchema, setInputSchema] = useState<SchemaField[]>([]);
  const [outputSchema, setOutputSchema] = useState<SchemaField[]>([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Update project type when prop changes
  useEffect(() => {
    setFormData(prev => ({ ...prev, project_type: projectType }));
  }, [projectType]);

  // Reset state when dialog opens/closes
  useEffect(() => {
    if (open) {
      setCurrentStep(0);
      setFormData({ name: '', description: '', project_type: projectType });
      setInputSchema([]);
      setOutputSchema([]);
      setError(null);
    }
  }, [open, projectType]);

  const handleNext = () => {
    if (currentStep === 0) {
      // Validate basic info
      if (!formData.name.trim()) {
        setError('Project name is required');
        return;
      }
    }

    setError(null);
    setCurrentStep(prev => Math.min(prev + 1, STEPS.length - 1));
  };

  const handleBack = () => {
    setError(null);
    setCurrentStep(prev => Math.max(prev - 1, 0));
  };

  const handleSkipStep = () => {
    setError(null);
    if (currentStep < STEPS.length - 1) {
      setCurrentStep(prev => prev + 1);
    } else {
      handleSubmit();
    }
  };

  // Dtypes (including List/Struct/Custom objects) go to the server unchanged
  const convertSchemaForAPI = (schema: SchemaField[]) => schema;

  const buildProjectData = () => {
    if (projectType === 'flat_rule') {
      // Build PrioritizedTreeModule structure
      return {
        type: 'prioritized_flat_rule',
        output: {
          data: [],
          dtypes: convertSchemaForAPI(outputSchema),
          type_defs: {},
          default: null
        },
        rules: [],
        parameters_col: 'parameters',
        parameters: {},
        mode: 'first_match',
        ...(inputSchema.length > 0 && { input_schema: convertSchemaForAPI(inputSchema) })
      };
    } else {
      // Build v3 Tree structure
      return {
        type: 'ui-tree',
        formatVersion: 3,
        metadata: {
          name: formData.name,
          description: formData.description || ''
        },
        nodes: [],
        edges: [],
        subtrees: [],
        output: {
          data: [],
          dtypes: convertSchemaForAPI(outputSchema) as any,
          type_defs: {},
          default: null
        },
        input_schema: convertSchemaForAPI(inputSchema) as any,
        parameters: {}
      };
    }
  };

  const handleSubmit = async () => {
    if (!formData.name.trim()) {
      setError('Project name is required');
      setCurrentStep(0);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const projectData = buildProjectData();

      const projectCreateData: ProjectCreate = {
        name: formData.name.trim(),
        description: formData.description.trim() || undefined,
        project_type: projectType,
        project_data: projectData as any // Type conversion handled by backend
      };

      const response = await createProjectApiV1ProjectsPost({
        body: projectCreateData
      });

      if (response.data) {
        // Success - reset and close
        setCurrentStep(0);
        setFormData({ name: '', description: '', project_type: projectType });
        setInputSchema([]);
        setOutputSchema([]);
        onOpenChange(false);
        onProjectCreated(response.data.id);

        toast({
          title: "Project Created",
          description: `Successfully created ${projectType === 'tree' ? 'tree' : 'flat rule'} project "${formData.name}"`,
        });
      }
    } catch (err: any) {
      console.error('Failed to create project:', err);

      // Extract meaningful error message
      let errorMessage = 'Failed to create project';

      if (err.detail && Array.isArray(err.detail)) {
        // Pydantic validation errors
        const errors = err.detail.map((e: any) => {
          const field = e.loc ? e.loc.join('.') : 'unknown';
          return `${field}: ${e.msg}`;
        }).join('; ');
        errorMessage = `Validation error: ${errors}`;
      } else if (typeof err.detail === 'string') {
        errorMessage = err.detail;
      } else if (err.message && typeof err.message === 'string') {
        errorMessage = err.message;
      } else if (typeof err === 'string') {
        errorMessage = err;
      }

      setError(errorMessage);

      toast({
        title: "Failed to create project",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setCurrentStep(0);
    setFormData({ name: '', description: '', project_type: projectType });
    setInputSchema([]);
    setOutputSchema([]);
    setError(null);
    onOpenChange(false);
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return (
          <BasicInfoStep
            formData={formData}
            setFormData={setFormData}
            loading={loading}
            projectType={projectType}
          />
        );
      case 1:
        return (
          <ConfigureInputsStep
            inputSchema={inputSchema}
            setInputSchema={setInputSchema}
            projectType={projectType}
          />
        );
      case 2:
        return (
          <ConfigureOutputsStep
            outputSchema={outputSchema}
            setOutputSchema={setOutputSchema}
            projectType={projectType}
          />
        );
      default:
        return null;
    }
  };

  const canProceed = () => {
    if (currentStep === 0) {
      return formData.name.trim().length > 0;
    }
    return true;
  };

  const projectIcon = projectType === 'tree' ? Network : FileCode;
  const projectLabel = projectType === 'tree' ? 'Tree' : 'Flat Rule';
  const iconColor = projectType === 'tree' ? 'text-blue-600' : 'text-green-600';

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {React.createElement(projectIcon, { className: `w-5 h-5 ${iconColor}` })}
            Create New {projectLabel} Project
          </DialogTitle>
        </DialogHeader>

        {/* Stepper */}
        <div className="py-4">
          <Stepper steps={STEPS} currentStep={currentStep} />
        </div>

        {/* Step Content */}
        <div className="min-h-[300px]">
          {renderStepContent()}
        </div>

        {/* Error Display */}
        {error && (
          <div className="text-red-600 text-sm bg-red-50 p-3 rounded-md">
            {error}
          </div>
        )}

        {/* Navigation Buttons */}
        <div className="flex justify-between pt-4 border-t">
          <Button
            type="button"
            variant="outline"
            onClick={currentStep === 0 ? handleCancel : handleBack}
            disabled={loading}
          >
            {currentStep === 0 ? (
              'Cancel'
            ) : (
              <>
                <ChevronLeft className="w-4 h-4 mr-1" />
                Back
              </>
            )}
          </Button>

          <div className="flex gap-2">
            {/* Skip button for steps 1 and 2 */}
            {currentStep > 0 && (
              <Button
                type="button"
                variant="ghost"
                onClick={handleSkipStep}
                disabled={loading}
              >
                Skip
              </Button>
            )}

            {currentStep < STEPS.length - 1 ? (
              <Button
                type="button"
                onClick={handleNext}
                disabled={loading || !canProceed()}
              >
                Next
                <ChevronRight className="w-4 h-4 ml-1" />
              </Button>
            ) : (
              <Button
                type="button"
                onClick={handleSubmit}
                disabled={loading || !canProceed()}
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Creating...
                  </>
                ) : (
                  'Create Project'
                )}
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

// Step 1: Basic Info
interface BasicInfoStepProps {
  formData: ProjectFormData;
  setFormData: React.Dispatch<React.SetStateAction<ProjectFormData>>;
  loading: boolean;
  projectType: ProjectType;
}

const BasicInfoStep: React.FC<BasicInfoStepProps> = ({ formData, setFormData, loading, projectType }) => {
  const projectLabel = projectType === 'tree' ? 'tree' : 'flat rule';

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="name">Project Name *</Label>
        <Input
          id="name"
          value={formData.name}
          onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
          placeholder="Enter project name"
          disabled={loading}
          className="mt-1"
          autoFocus
        />
      </div>

      <div>
        <Label htmlFor="description">Description</Label>
        <Textarea
          id="description"
          value={formData.description || ''}
          onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
          placeholder={`Describe your ${projectLabel} project (optional)`}
          disabled={loading}
          className="mt-1"
          rows={4}
        />
      </div>
    </div>
  );
};

// Step 2: Configure Inputs
interface ConfigureInputsStepProps {
  inputSchema: SchemaField[];
  setInputSchema: React.Dispatch<React.SetStateAction<SchemaField[]>>;
  projectType: ProjectType;
}

const ConfigureInputsStep: React.FC<ConfigureInputsStepProps> = ({
  inputSchema,
  setInputSchema,
  projectType
}) => {
  const [showLoadDialog, setShowLoadDialog] = useState(false);

  const handleSchemaLoaded = (newSchema: SchemaField[]) => {
    setInputSchema(newSchema);
    setShowLoadDialog(false);
  };

  const fieldLabel = projectType === 'tree' ? 'Features' : 'Input Fields';

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Database className="h-5 w-5" />
            {fieldLabel}
          </h3>
          <p className="text-sm text-gray-600 mt-1">
            {projectType === 'tree'
              ? 'Define the features (input fields) for your decision tree'
              : 'Define the input fields for your decision tree'}
          </p>
        </div>
      </div>

      <SchemaFieldList
        fields={inputSchema}
        onChange={setInputSchema}
        showLoadFromDataset={true}
        onLoadFromDataset={() => setShowLoadDialog(true)}
        showSearch={true}
        emptyStateTitle={`No ${fieldLabel.toLowerCase()} defined yet`}
        emptyStateDescription="Add fields or load from dataset"
        addButtonLabel="Add Field"
        maxHeight="300px"
      />

      <LoadFromDatasetDialog
        open={showLoadDialog}
        onClose={() => setShowLoadDialog(false)}
        onSchemaLoaded={handleSchemaLoaded}
      />
    </div>
  );
};

// Step 3: Configure Outputs
interface ConfigureOutputsStepProps {
  outputSchema: SchemaField[];
  setOutputSchema: React.Dispatch<React.SetStateAction<SchemaField[]>>;
  projectType: ProjectType;
}

const ConfigureOutputsStep: React.FC<ConfigureOutputsStepProps> = ({
  outputSchema,
  setOutputSchema,
  projectType
}) => {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Database className="h-5 w-5" />
            Output Schema
          </h3>
          <p className="text-sm text-gray-600 mt-1">
            Define the output fields for your decision tree results (optional)
          </p>
        </div>
      </div>

      <SchemaFieldList
        fields={outputSchema}
        onChange={setOutputSchema}
        showLoadFromDataset={false}
        showSearch={true}
        emptyStateTitle="No output fields defined yet"
        emptyStateDescription="Add fields to define your output schema"
        addButtonLabel="Add Field"
        maxHeight="300px"
      />
    </div>
  );
};
