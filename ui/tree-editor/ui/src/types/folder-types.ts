// Re-export types from generated API for consistency
import type {
  AccessMode as ApiAccessMode,
  ShareType as ApiShareType,
  FolderResponse,
  FolderShareResponse,
} from '@/api/types.gen';

export type AccessMode = ApiAccessMode;
export type ShareType = ApiShareType;

// Use the generated FolderResponse as the main Folder type
export type Folder = FolderResponse;

export interface FolderWithChildren extends Folder {
  children: Folder[];
  projects: Array<{
    id: string;
    name: string;
    project_type: 'tree' | 'flat_rule';
  }>;
}

export interface FolderBreadcrumbItem {
  id: string;
  name: string;
}

// Use the generated FolderShareResponse
export type FolderShare = FolderShareResponse;

export interface EffectivePermissions {
  access_mode: AccessMode;
  is_owner: boolean;
  inherited_from?: string; // Folder ID if inherited
  permission_sources: Array<{
    source_type: 'ownership' | 'direct_share' | 'folder_share' | 'team_share';
    source_id?: string;
    source_name?: string;
    access_mode: AccessMode;
  }>;
}

// API Request/Response types

export interface FolderCreate {
  name: string;
  description?: string;
  parent_id?: string;
}

export interface FolderUpdate {
  name?: string;
  description?: string;
  parent_id?: string;
}

export interface FolderMove {
  new_parent_id?: string; // null for root
}

export interface FolderShareCreate {
  share_type: ShareType;
  user_id?: string;
  team_id?: string;
  access_mode: AccessMode;
}

export interface DeleteFolderOptions {
  move_to_root?: boolean;
  delete_projects?: boolean;
}

// Mock data for development
export const mockFolders: Folder[] = [
  {
    id: '1',
    name: 'Marketing',
    description: 'Marketing decision trees and campaigns',
    parent_id: null,
    created_by: 'user-1',
    created_at: '2024-01-15T10:00:00Z',
    updated_at: '2024-01-20T14:30:00Z',
    access_mode: 'admin',
    project_count: 5,
    subfolder_count: 2,
    is_shared: true,
    is_owner: true,
  },
  {
    id: '2',
    name: 'Q1 2024',
    description: 'First quarter campaigns',
    parent_id: '1',
    created_by: 'user-1',
    created_at: '2024-01-16T10:00:00Z',
    updated_at: '2024-01-18T14:30:00Z',
    access_mode: 'admin',
    project_count: 3,
    subfolder_count: 0,
    is_shared: false,
    is_owner: true,
  },
  {
    id: '3',
    name: 'Q2 2024',
    description: 'Second quarter campaigns',
    parent_id: '1',
    created_by: 'user-1',
    created_at: '2024-04-01T10:00:00Z',
    updated_at: '2024-04-05T14:30:00Z',
    access_mode: 'admin',
    project_count: 2,
    subfolder_count: 0,
    is_shared: false,
    is_owner: true,
  },
  {
    id: '4',
    name: 'Risk Models',
    description: 'Credit and fraud risk assessment models',
    parent_id: null,
    created_by: 'user-1',
    created_at: '2024-02-01T10:00:00Z',
    updated_at: '2024-02-10T14:30:00Z',
    access_mode: 'write',
    project_count: 8,
    subfolder_count: 1,
    is_shared: true,
    is_owner: false,
  },
  {
    id: '5',
    name: 'Production Models',
    description: 'Live production risk models',
    parent_id: '4',
    created_by: 'user-2',
    created_at: '2024-02-05T10:00:00Z',
    updated_at: '2024-02-15T14:30:00Z',
    access_mode: 'read',
    project_count: 5,
    subfolder_count: 0,
    is_shared: true,
    is_owner: false,
  },
];
