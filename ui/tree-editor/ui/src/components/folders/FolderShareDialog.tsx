import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Search, Users, User as UserIcon, Trash2, Info, ArrowDown } from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import { useFolders } from '@/contexts/FolderContext';
import { searchUsersApiV1UsersSearchGet, searchTeamsApiV1TeamsSearchGet } from '@/api';
import type { UserSearchResult } from '@/api/types.gen';
import type { Folder, FolderShare, AccessMode, ShareType } from '@/types/folder-types';

// Team search result type (backend doesn't have proper schema yet)
interface TeamSearchResult {
  id: string;
  name: string;
  member_count?: number;
}

interface FolderShareDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  folder: Folder | null;
  onShareAdded?: () => void;
}

const ACCESS_MODES: Array<{ value: AccessMode; label: string; description: string }> = [
  { value: 'read', label: 'Read Only', description: 'Can view folder and projects' },
  { value: 'write', label: 'Read & Write', description: 'Can edit projects and create new ones' },
  { value: 'admin', label: 'Admin', description: 'Full control including sharing and deletion' },
];

export const FolderShareDialog: React.FC<FolderShareDialogProps> = ({
  open,
  onOpenChange,
  folder,
  onShareAdded,
}) => {
  const { getFolderShares, createFolderShare, removeFolderShare, getEffectivePermissions } = useFolders();
  const [activeTab, setActiveTab] = useState<'add' | 'manage'>('add');
  const [shareType, setShareType] = useState<ShareType>('user');
  const [searchQuery, setSearchQuery] = useState('');
  const [accessMode, setAccessMode] = useState<AccessMode>('read');
  const [searchResults, setSearchResults] = useState<(UserSearchResult | TeamSearchResult)[]>([]);
  const [existingShares, setExistingShares] = useState<FolderShare[]>([]);
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);

  // Real API search functions
  const searchUsers = async (query: string): Promise<UserSearchResult[]> => {
    const response = await searchUsersApiV1UsersSearchGet({
      query: { q: query },
    });
    return response.data?.users || [];
  };

  const searchTeams = async (query: string): Promise<TeamSearchResult[]> => {
    const response = await searchTeamsApiV1TeamsSearchGet({
      query: { q: query },
    });
    // Backend returns unknown type, cast to expected structure
    const data = response.data as { teams?: TeamSearchResult[] } | undefined;
    return data?.teams || [];
  };

  // Load existing shares when dialog opens
  useEffect(() => {
    if (open && folder) {
      loadExistingShares();
    }
  }, [open, folder]);

  // Search users/teams when query changes
  useEffect(() => {
    if (searchQuery.length >= 2) {
      searchUsersOrTeams();
    } else {
      setSearchResults([]);
    }
  }, [searchQuery, shareType]);

  const loadExistingShares = async () => {
    if (!folder) return;

    try {
      const shares = await getFolderShares(folder.id);
      setExistingShares(shares);
    } catch (error) {
      console.error('Failed to load folder shares:', error);
    }
  };

  const searchUsersOrTeams = async () => {
    setSearching(true);
    try {
      if (shareType === 'user') {
        const users = await searchUsers(searchQuery);
        setSearchResults(users);
      } else {
        const teams = await searchTeams(searchQuery);
        setSearchResults(teams);
      }
    } catch (error) {
      console.error('Search failed:', error);
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  };

  const handleAddShare = async (targetId: string) => {
    if (!folder) return;

    setLoading(true);
    try {
      await createFolderShare(folder.id, {
        share_type: shareType,
        user_id: shareType === 'user' ? targetId : undefined,
        team_id: shareType === 'team' ? targetId : undefined,
        access_mode: accessMode,
      });

      toast({
        title: 'Share added',
        description: 'Folder has been shared successfully.',
      });

      // Refresh shares list
      await loadExistingShares();
      onShareAdded?.();

      // Reset form
      setSearchQuery('');
      setSearchResults([]);
      setAccessMode('read');
    } catch (error: any) {
      console.error('Failed to add share:', error);
      toast({
        title: 'Share failed',
        description: error.message || 'Failed to share folder. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveShare = async (shareId: string) => {
    if (!folder) return;

    setLoading(true);
    try {
      await removeFolderShare(folder.id, shareId);

      toast({
        title: 'Share removed',
        description: 'Share has been removed successfully.',
      });

      // Refresh shares list
      await loadExistingShares();
      onShareAdded?.();
    } catch (error: any) {
      console.error('Failed to remove share:', error);
      toast({
        title: 'Remove failed',
        description: error.message || 'Failed to remove share. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  if (!folder) return null;

  const hasProjects = (folder.project_count || 0) > 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Share Folder: {folder.name}</DialogTitle>
          <DialogDescription>Manage who can access this folder and its contents</DialogDescription>
        </DialogHeader>

        {hasProjects && (
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription className="flex items-start gap-2">
              <ArrowDown className="h-4 w-4 flex-shrink-0 mt-0.5" />
              <span>
                Sharing this folder will grant access to all {folder.project_count} project{folder.project_count !== 1 ? 's' : ''} inside it. Permissions will inherit to subfolders and projects.
              </span>
            </AlertDescription>
          </Alert>
        )}

        <Tabs value={activeTab} onValueChange={(val) => setActiveTab(val as 'add' | 'manage')}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="add">Add Share</TabsTrigger>
            <TabsTrigger value="manage">
              Manage Shares
              {existingShares.length > 0 && (
                <Badge variant="secondary" className="ml-2">
                  {existingShares.length}
                </Badge>
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="add" className="space-y-4">
            <div className="space-y-2">
              <Label>Share with</Label>
              <div className="flex gap-2">
                <Button
                  variant={shareType === 'user' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setShareType('user')}
                  className="flex-1"
                >
                  <UserIcon className="h-4 w-4 mr-2" />
                  User
                </Button>
                <Button
                  variant={shareType === 'team' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setShareType('team')}
                  className="flex-1"
                >
                  <Users className="h-4 w-4 mr-2" />
                  Team
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="search">{shareType === 'user' ? 'Search Users' : 'Search Teams'}</Label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  id="search"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder={shareType === 'user' ? 'Enter email or name...' : 'Enter team name...'}
                  className="pl-10"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="access-mode">Access Level</Label>
              <Select value={accessMode} onValueChange={(val) => setAccessMode(val as AccessMode)}>
                <SelectTrigger id="access-mode">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ACCESS_MODES.map(mode => (
                    <SelectItem key={mode.value} value={mode.value}>
                      <div className="flex flex-col">
                        <span className="font-medium">{mode.label}</span>
                        <span className="text-xs text-gray-500">{mode.description}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {searching && (
              <div className="text-sm text-gray-500 text-center py-4">Searching...</div>
            )}

            {searchResults.length > 0 && (
              <div className="border rounded-md divide-y max-h-60 overflow-y-auto">
                {searchResults.map(result => {
                  const isUser = 'email' in result;
                  return (
                    <div key={result.id} className="flex items-center justify-between p-3 hover:bg-gray-50">
                      <div className="flex items-center gap-3">
                        {isUser ? (
                          <UserIcon className="h-5 w-5 text-gray-400" />
                        ) : (
                          <Users className="h-5 w-5 text-gray-400" />
                        )}
                        <div>
                          <div className="font-medium">
                            {isUser ? (result as UserSearchResult).full_name || (result as UserSearchResult).email : (result as TeamSearchResult).name}
                          </div>
                          {isUser && (result as UserSearchResult).full_name && (
                            <div className="text-sm text-gray-500">{(result as UserSearchResult).email}</div>
                          )}
                          {!isUser && (result as TeamSearchResult).member_count && (
                            <div className="text-sm text-gray-500">{(result as TeamSearchResult).member_count} members</div>
                          )}
                        </div>
                      </div>
                      <Button size="sm" onClick={() => handleAddShare(result.id)} disabled={loading}>
                        Share
                      </Button>
                    </div>
                  );
                })}
              </div>
            )}

            {searchQuery.length >= 2 && !searching && searchResults.length === 0 && (
              <div className="text-sm text-gray-500 text-center py-4">No results found</div>
            )}
          </TabsContent>

          <TabsContent value="manage" className="space-y-4">
            {existingShares.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Users className="h-8 w-8 mx-auto mb-2 text-gray-400" />
                <p>This folder is not shared yet</p>
                <Button variant="link" onClick={() => setActiveTab('add')} className="mt-2">
                  Add your first share
                </Button>
              </div>
            ) : (
              <div className="border rounded-md divide-y">
                {existingShares.map(share => {
                  const isUser = share.share_type === 'user';
                  // Backend provides user_email and team_name in the response
                  const displayName = isUser
                    ? share.user_email || 'Unknown User'
                    : share.team_name || 'Unknown Team';
                  // For users, show email as primary if no full name available
                  const displaySubtext = isUser && share.user_email ? share.user_email : undefined;

                  return (
                    <div key={share.id} data-testid="share-row" className="flex items-center justify-between p-3">
                      <div className="flex items-center gap-3 flex-1">
                        {isUser ? (
                          <UserIcon className="h-5 w-5 text-gray-400 flex-shrink-0" />
                        ) : (
                          <Users className="h-5 w-5 text-gray-400 flex-shrink-0" />
                        )}
                        <div className="flex-1 min-w-0">
                          <div className="font-medium truncate">{displayName}</div>
                          {displaySubtext && displayName !== displaySubtext && (
                            <div className="text-sm text-gray-500 truncate">{displaySubtext}</div>
                          )}
                        </div>
                        <Badge variant={share.access_mode === 'admin' ? 'default' : share.access_mode === 'write' ? 'secondary' : 'outline'}>
                          {ACCESS_MODES.find(m => m.value === share.access_mode)?.label || share.access_mode}
                        </Badge>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleRemoveShare(share.id)}
                        disabled={loading}
                        data-testid="share-remove"
                        className="ml-2 text-red-600 hover:text-red-700 hover:bg-red-50"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  );
                })}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
};
