import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import {
  listFoldersApiV1FoldersGet,
  createFolderApiV1FoldersPost,
  updateFolderApiV1FoldersFolderIdPut,
  deleteFolderApiV1FoldersFolderIdDelete,
  moveFolderApiV1FoldersFolderIdMovePost,
  getFolderApiV1FoldersFolderIdGet,
  listFolderSharesApiV1FoldersFolderIdSharesGet,
  createFolderShareApiV1FoldersFolderIdSharesPost,
  deleteFolderShareApiV1FoldersFolderIdSharesShareIdDelete,
  getFolderEffectivePermissionsApiV1FoldersFolderIdEffectivePermissionsGet,
} from '@/api';
import type {
  FolderResponse,
  FolderDetailResponse,
  FolderCreate,
  FolderUpdate,
  FolderMove,
  FolderShareResponse,
  FolderShareCreate,
  EffectivePermissions,
  AccessMode,
} from '@/api/types.gen';

// Type aliases for consistency with UI types
type Folder = FolderResponse;
type FolderWithChildren = FolderDetailResponse;
type FolderShare = FolderShareResponse;

interface DeleteFolderOptions {
  move_to_root?: boolean;
  delete_projects?: boolean;
}

interface FolderContextType {
  // State
  currentFolder: Folder | null;
  folderHierarchy: Folder[]; // Breadcrumb path from root to current
  folders: Folder[]; // All accessible folders (flat list)
  loading: boolean;
  error: string | null;

  // Navigation
  navigateToFolder: (folderId: string | null) => Promise<void>;
  navigateToRoot: () => void;

  // CRUD
  createFolder: (data: FolderCreate) => Promise<Folder>;
  updateFolder: (id: string, data: FolderUpdate) => Promise<Folder>;
  deleteFolder: (id: string, options: DeleteFolderOptions) => Promise<void>;
  moveFolder: (id: string, move: FolderMove) => Promise<Folder>;
  getFolderDetails: (id: string) => Promise<FolderWithChildren>;

  // Sharing
  getFolderShares: (folderId: string) => Promise<FolderShare[]>;
  createFolderShare: (folderId: string, share: FolderShareCreate) => Promise<FolderShare>;
  removeFolderShare: (folderId: string, shareId: string) => Promise<void>;

  // Permissions
  getEffectivePermissions: (folderId: string) => Promise<EffectivePermissions>;

  // Utilities
  refreshFolders: () => Promise<void>;
  buildFolderTree: () => Folder[];
}

const FolderContext = createContext<FolderContextType | undefined>(undefined);

export const useFolders = () => {
  const context = useContext(FolderContext);
  if (!context) {
    throw new Error('useFolders must be used within a FolderProvider');
  }
  return context;
};

interface FolderProviderProps {
  children: React.ReactNode;
}

export const FolderProvider: React.FC<FolderProviderProps> = ({ children }) => {
  const [currentFolder, setCurrentFolder] = useState<Folder | null>(null);
  const [folderHierarchy, setFolderHierarchy] = useState<Folder[]>([]);
  const [folders, setFolders] = useState<Folder[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Real API functions
  const api = {
    async listFolders(): Promise<Folder[]> {
      const response = await listFoldersApiV1FoldersGet();
      return response.data || [];
    },

    async getFolderDetails(id: string): Promise<FolderWithChildren> {
      const response = await getFolderApiV1FoldersFolderIdGet({
        path: { folder_id: id },
      });
      if (!response.data) throw new Error('Folder not found');
      return response.data;
    },

    async createFolder(data: FolderCreate): Promise<Folder> {
      const response = await createFolderApiV1FoldersPost({
        body: data,
      });
      if (!response.data) throw new Error('Failed to create folder');
      return response.data;
    },

    async updateFolder(id: string, data: FolderUpdate): Promise<Folder> {
      const response = await updateFolderApiV1FoldersFolderIdPut({
        path: { folder_id: id },
        body: data,
      });
      if (!response.data) throw new Error('Failed to update folder');
      return response.data;
    },

    async deleteFolder(id: string, options: DeleteFolderOptions): Promise<void> {
      await deleteFolderApiV1FoldersFolderIdDelete({
        path: { folder_id: id },
        query: options,
      });
    },

    async moveFolder(id: string, move: FolderMove): Promise<Folder> {
      const response = await moveFolderApiV1FoldersFolderIdMovePost({
        path: { folder_id: id },
        body: move,
      });
      if (!response.data) throw new Error('Failed to move folder');
      return response.data;
    },

    async getFolderShares(folderId: string): Promise<FolderShare[]> {
      const response = await listFolderSharesApiV1FoldersFolderIdSharesGet({
        path: { folder_id: folderId },
      });
      return response.data || [];
    },

    async createFolderShare(folderId: string, share: FolderShareCreate): Promise<FolderShare> {
      const response = await createFolderShareApiV1FoldersFolderIdSharesPost({
        path: { folder_id: folderId },
        body: share,
      });
      if (!response.data) throw new Error('Failed to create share');
      return response.data;
    },

    async removeFolderShare(folderId: string, shareId: string): Promise<void> {
      await deleteFolderShareApiV1FoldersFolderIdSharesShareIdDelete({
        path: {
          folder_id: folderId,
          share_id: shareId,
        },
      });
    },

    async getEffectivePermissions(folderId: string): Promise<EffectivePermissions> {
      const response = await getFolderEffectivePermissionsApiV1FoldersFolderIdEffectivePermissionsGet({
        path: { folder_id: folderId },
      });
      if (!response.data) throw new Error('Failed to get permissions');
      return response.data;
    },
  };

  // Build folder hierarchy (breadcrumb) from current folder to root
  const buildHierarchy = useCallback((folderId: string | null, allFolders: Folder[]): Folder[] => {
    if (!folderId) return [];

    const hierarchy: Folder[] = [];
    let currentId: string | null = folderId;

    while (currentId) {
      const folder = allFolders.find(f => f.id === currentId);
      if (!folder) break;

      hierarchy.unshift(folder); // Add to beginning
      currentId = folder.parent_id ?? null;
    }

    return hierarchy;
  }, []);

  // Build folder tree structure (hierarchical)
  const buildFolderTree = useCallback((): Folder[] => {
    const rootFolders = folders.filter(f => !f.parent_id);
    return rootFolders;
  }, [folders]);

  // Fetch all folders
  const refreshFolders = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const allFolders = await api.listFolders();
      setFolders(allFolders);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load folders';
      setError(message);
      console.error('Failed to fetch folders:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Navigate to a specific folder
  const navigateToFolder = useCallback(async (folderId: string | null) => {
    try {
      setLoading(true);
      setError(null);

      if (folderId === null) {
        setCurrentFolder(null);
        setFolderHierarchy([]);
        return;
      }

      const folder = folders.find(f => f.id === folderId);
      if (!folder) {
        throw new Error('Folder not found');
      }

      setCurrentFolder(folder);
      const hierarchy = buildHierarchy(folderId, folders);
      setFolderHierarchy(hierarchy);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to navigate to folder';
      setError(message);
      console.error('Failed to navigate:', err);
    } finally {
      setLoading(false);
    }
  }, [folders, buildHierarchy]);

  // Navigate to root
  const navigateToRoot = useCallback(() => {
    setCurrentFolder(null);
    setFolderHierarchy([]);
  }, []);

  // Create folder
  const createFolder = useCallback(async (data: FolderCreate): Promise<Folder> => {
    try {
      setLoading(true);
      setError(null);
      const newFolder = await api.createFolder(data);

      // Add to local state
      setFolders(prev => [...prev, newFolder]);

      return newFolder;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create folder';
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Update folder
  const updateFolder = useCallback(async (id: string, data: FolderUpdate): Promise<Folder> => {
    try {
      setLoading(true);
      setError(null);
      const updatedFolder = await api.updateFolder(id, data);

      // Update in local state
      setFolders(prev => prev.map(f => f.id === id ? updatedFolder : f));

      if (currentFolder?.id === id) {
        setCurrentFolder(updatedFolder);
      }

      return updatedFolder;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to update folder';
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [currentFolder]);

  // Delete folder
  const deleteFolder = useCallback(async (id: string, options: DeleteFolderOptions): Promise<void> => {
    try {
      setLoading(true);
      setError(null);
      await api.deleteFolder(id, options);

      // Remove from local state
      setFolders(prev => prev.filter(f => f.id !== id));

      if (currentFolder?.id === id) {
        setCurrentFolder(null);
        setFolderHierarchy([]);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete folder';
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [currentFolder]);

  // Move folder
  const moveFolder = useCallback(async (id: string, move: FolderMove): Promise<Folder> => {
    try {
      setLoading(true);
      setError(null);
      const movedFolder = await api.moveFolder(id, move);

      // Update in local state
      setFolders(prev => prev.map(f => f.id === id ? movedFolder : f));

      return movedFolder;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to move folder';
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Get folder details
  const getFolderDetails = useCallback(async (id: string): Promise<FolderWithChildren> => {
    try {
      return await api.getFolderDetails(id);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to get folder details';
      setError(message);
      throw err;
    }
  }, []);

  // Sharing functions
  const getFolderShares = useCallback(async (folderId: string): Promise<FolderShare[]> => {
    return await api.getFolderShares(folderId);
  }, []);

  const createFolderShare = useCallback(async (folderId: string, share: FolderShareCreate): Promise<FolderShare> => {
    return await api.createFolderShare(folderId, share);
  }, []);

  const removeFolderShare = useCallback(async (folderId: string, shareId: string): Promise<void> => {
    await api.removeFolderShare(folderId, shareId);
  }, []);

  const getEffectivePermissions = useCallback(async (folderId: string): Promise<EffectivePermissions> => {
    return await api.getEffectivePermissions(folderId);
  }, []);

  // Load folders on mount
  useEffect(() => {
    refreshFolders();
  }, []);

  const value: FolderContextType = {
    currentFolder,
    folderHierarchy,
    folders,
    loading,
    error,
    navigateToFolder,
    navigateToRoot,
    createFolder,
    updateFolder,
    deleteFolder,
    moveFolder,
    getFolderDetails,
    getFolderShares,
    createFolderShare,
    removeFolderShare,
    getEffectivePermissions,
    refreshFolders,
    buildFolderTree,
  };

  return (
    <FolderContext.Provider value={value}>
      {children}
    </FolderContext.Provider>
  );
};
