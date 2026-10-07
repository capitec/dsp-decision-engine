import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Search, Users, User as UserIcon, Trash2, Edit } from "lucide-react";
import {
  getProjectSharesApiV1ProjectsProjectIdSharesGet,
  searchUsersApiV1UsersSearchGet,
  searchTeamsApiV1TeamsSearchGet,
  shareProjectApiV1ProjectsProjectIdSharesPost,
  removeProjectShareApiV1ProjectsProjectIdSharesShareIdDelete
} from "@/api";
import type { UserSearchResult, ProjectShareResponse, AccessMode } from "@/api/types.gen";

// Team search result type (backend doesn't have proper schema yet)
interface TeamSearchResult {
  id: string;
  name: string;
  member_count?: number;
}

interface Project {
  id: string;
  name: string;
}

interface ShareProjectDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  project: Project | null;
  onShareAdded: () => void;
}

const ACCESS_MODES = [
  { value: 'read', label: 'Read Only' },
  { value: 'write', label: 'Read & Write' },
  { value: 'admin', label: 'Admin' }
];

export const ShareProjectDialog: React.FC<ShareProjectDialogProps> = ({
  open,
  onOpenChange,
  project,
  onShareAdded
}) => {
  const [activeTab, setActiveTab] = useState<'add' | 'manage'>('add');
  const [shareType, setShareType] = useState<'user' | 'team'>('user');
  const [searchQuery, setSearchQuery] = useState('');
  const [accessMode, setAccessMode] = useState<AccessMode>('read');
  const [searchResults, setSearchResults] = useState<(UserSearchResult | TeamSearchResult)[]>([]);
  const [existingShares, setExistingShares] = useState<ProjectShareResponse[]>([]);
  const [selectedShare, setSelectedShare] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);

  // Load existing shares when dialog opens
  useEffect(() => {
    if (open && project) {
      loadExistingShares();
    }
  }, [open, project]);

  // Search users/teams when query changes
  useEffect(() => {
    if (searchQuery.length >= 2) {
      searchUsersOrTeams();
    } else {
      setSearchResults([]);
    }
  }, [searchQuery, shareType]);

  const loadExistingShares = async () => {
    if (!project) return;

    try {
      const response = await getProjectSharesApiV1ProjectsProjectIdSharesGet({
        path: { project_id: project.id }
      });
      if (response.data) {
        setExistingShares(response.data);
      }
    } catch (error) {
      console.error('Failed to load project shares:', error);
    }
  };

  const searchUsersOrTeams = async () => {
    setSearching(true);
    try {
      if (shareType === 'user') {
        const response = await searchUsersApiV1UsersSearchGet({
          query: { q: searchQuery }
        });
        setSearchResults(response.data?.users || []);
      } else {
        const response = await searchTeamsApiV1TeamsSearchGet({
          query: { q: searchQuery }
        });
        // Backend returns unknown type, cast to expected structure
        const data = response.data as { teams?: TeamSearchResult[] } | undefined;
        setSearchResults(data?.teams || []);
      }
    } catch (error) {
      console.error('Search failed:', error);
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  };

  const handleAddShare = async (targetId: string) => {
    if (!project) return;

    setLoading(true);
    try {
      await shareProjectApiV1ProjectsProjectIdSharesPost({
        path: { project_id: project.id },
        body: {
          share_type: shareType,
          user_id: shareType === 'user' ? targetId : undefined,
          team_id: shareType === 'team' ? targetId : undefined,
          access_mode: accessMode
        }
      });

      // Refresh shares list
      await loadExistingShares();
      onShareAdded();

      // Reset form
      setSearchQuery('');
      setSearchResults([]);
      setAccessMode('read');

    } catch (error) {
      console.error('Failed to add share:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateShare = async (shareId: string, newAccessMode: AccessMode) => {
    // Note: Update endpoint doesn't exist in the generated API
    // This functionality may need to be implemented on the backend
    console.warn('Update share endpoint not available in API');
  };

  const handleRemoveShare = async (shareId: string) => {
    if (!project) return;

    setLoading(true);
    try {
      await removeProjectShareApiV1ProjectsProjectIdSharesShareIdDelete({
        path: {
          project_id: project.id,
          share_id: shareId
        }
      });

      await loadExistingShares();
      onShareAdded();
    } catch (error) {
      console.error('Failed to remove share:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!project) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Share Project: {project.name}</DialogTitle>
          <DialogDescription>
            Share this project with users or teams and manage their access levels.
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as 'add' | 'manage')} className="flex-1 overflow-hidden">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="add">Add Share</TabsTrigger>
            <TabsTrigger value="manage">Manage Shares ({existingShares.length})</TabsTrigger>
          </TabsList>

          <TabsContent value="add" className="space-y-4 overflow-auto">
            <div>
              <Label>Share with</Label>
              <Tabs value={shareType} onValueChange={(value) => setShareType(value as 'user' | 'team')}>
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="user">
                    <UserIcon className="w-4 h-4 mr-2" />
                    Users
                  </TabsTrigger>
                  <TabsTrigger value="team">
                    <Users className="w-4 h-4 mr-2" />
                    Teams
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            <div>
              <Label>Search {shareType === 'user' ? 'Users' : 'Teams'}</Label>
              <div className="relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder={`Search ${shareType === 'user' ? 'by email' : 'by team name'}...`}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>

            <div>
              <Label>Access Level</Label>
              <Select value={accessMode} onValueChange={(value) => setAccessMode(value as any)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ACCESS_MODES.map((mode) => (
                    <SelectItem key={mode.value} value={mode.value}>
                      {mode.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {searching && (
              <div className="text-center py-4 text-muted-foreground">
                Searching...
              </div>
            )}

            {searchResults.length > 0 && (
              <div className="space-y-2 max-h-60 overflow-auto">
                <Label>Search Results</Label>
                {searchResults.map((result) => (
                  <div key={result.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div>
                      <div className="font-medium">
                        {shareType === 'user' ? (result as UserSearchResult).email : (result as TeamSearchResult).name}
                      </div>
                      {shareType === 'user' && (result as UserSearchResult).full_name && (
                        <div className="text-sm text-muted-foreground">{(result as UserSearchResult).full_name}</div>
                      )}
                      {shareType === 'team' && (result as TeamSearchResult).member_count !== undefined && (
                        <div className="text-sm text-muted-foreground">{(result as TeamSearchResult).member_count} members</div>
                      )}
                    </div>
                    <Button
                      onClick={() => handleAddShare(result.id)}
                      disabled={loading}
                      size="sm"
                    >
                      Share
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="manage" className="space-y-4 overflow-auto">
            {existingShares.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No shares found. Add some shares to get started.
              </div>
            ) : (
              <div className="space-y-3">
                {existingShares.map((share) => (
                  <div key={share.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      {share.share_type === 'user' ? (
                        <UserIcon className="w-5 h-5 text-muted-foreground" />
                      ) : (
                        <Users className="w-5 h-5 text-muted-foreground" />
                      )}
                      <div>
                        <div className="font-medium">
                          {share.share_type === 'user' ? share.user_email : share.team_name}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {share.share_type === 'user' ? 'User' : 'Team'} • 
                          Added {new Date(share.created_at).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Select
                        value={share.access_mode}
                        onValueChange={(value) => handleUpdateShare(share.id, value as any)}
                        disabled={loading}
                      >
                        <SelectTrigger className="w-32">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {ACCESS_MODES.map((mode) => (
                            <SelectItem key={mode.value} value={mode.value}>
                              {mode.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleRemoveShare(share.id)}
                        disabled={loading}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>

        <div className="flex justify-end pt-4">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};