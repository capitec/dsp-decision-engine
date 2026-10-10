// API service functions for flat rule system (now using unified project API)
import {
  listProjectsApiV1ProjectsGet,
  getProjectApiV1ProjectsProjectIdGet,
  createProjectApiV1ProjectsPost,
  updateProjectApiV1ProjectsProjectIdPut,
  deleteProjectApiV1ProjectsProjectIdDelete,
  exportProjectApiV1ProjectsProjectIdExportGet
} from '@/api';
import type { ProjectDetailResponse, ProjectResponse } from '@/api/types.gen';
import type {
  FlatRuleProject,
  FlatRuleProjectDetail,
  PaginatedResponse
} from './types';

// Flat rules are now just projects with project_type='flat_rule'
// These helper functions provide backward compatibility for the flat rule UI

function mapProjectToFlatRule(project: ProjectResponse): FlatRuleProject {
  return {
    id: project.id,
    name: project.name,
    description: project.description || undefined,
    created_at: project.created_at,
    updated_at: project.updated_at,
    version: project.version,
    created_by: project.created_by
  };
}

function mapProjectDetailToFlatRuleDetail(project: ProjectDetailResponse): FlatRuleProjectDetail {
  const projectData = project.project_data as any;

  return {
    id: project.id,
    name: project.name,
    description: project.description || undefined,
    created_at: project.created_at,
    updated_at: project.updated_at,
    version: project.version,
    created_by: project.created_by,
    project_data: projectData,
    validation_errors: project.validation_errors as any[]
  };
}

// Project operations
export const flatRuleApi = {
  // Projects
  listProjects: async (page = 1, size = 20): Promise<PaginatedResponse<FlatRuleProject>> => {
    const response = await listProjectsApiV1ProjectsGet({
      query: {
        page,
        size,
        project_type: 'flat_rule'
      }
    });

    if (!response.data) {
      throw new Error('Failed to load projects');
    }

    return {
      items: response.data.items.map(mapProjectToFlatRule),
      total: response.data.total || 0,
      page: response.data.page || 1,
      size: response.data.size || size,
      pages: response.data.pages || 1
    };
  },

  getProject: async (projectId: string): Promise<FlatRuleProjectDetail> => {
    const response = await getProjectApiV1ProjectsProjectIdGet({
      path: { project_id: projectId }
    });

    if (!response.data) {
      throw new Error('Failed to load project');
    }

    return mapProjectDetailToFlatRuleDetail(response.data);
  },

  createProject: async (data: any): Promise<FlatRuleProject> => {
    const response = await createProjectApiV1ProjectsPost({
      body: {
        name: data.name,
        description: data.description,
        project_type: 'flat_rule',
        project_data: data.project_data
      }
    });

    if (!response.data) {
      throw new Error('Failed to create project');
    }

    return mapProjectToFlatRule(response.data);
  },

  updateProject: async (projectId: string, data: any): Promise<FlatRuleProject & { validation_errors: any[] }> => {
    const response = await updateProjectApiV1ProjectsProjectIdPut({
      path: { project_id: projectId },
      body: {
        name: data.name,
        description: data.description,
        project_data: data.project_data,
        base_version: data.base_version
      }
    });

    if (!response.data) {
      throw new Error('Failed to update project');
    }

    return { ...mapProjectToFlatRule(response.data), validation_errors: response.data.validation_errors };
  },

  deleteProject: async (projectId: string): Promise<void> => {
    await deleteProjectApiV1ProjectsProjectIdDelete({
      path: { project_id: projectId }
    });
  },

  // Note: Individual rule/output/parameter CRUD operations are not used by the UI
  // The flat rule editor updates the entire project_data when making changes

  // Export
  exportProject: async (projectId: string, _options: any = {}): Promise<any> => {
    const response = await exportProjectApiV1ProjectsProjectIdExportGet({
      path: { project_id: projectId }
    });
    return response.data;
  },

  // Download export as file (an invalid document comes back as a 400 and throws)
  downloadExport: async (projectId: string, projectName: string): Promise<void> => {
    const data = await flatRuleApi.exportProject(projectId);
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    link.download = `${projectName.replace(/[^a-zA-Z0-9]/g, '_')}_flat_rules_export.json`;

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  }

};
