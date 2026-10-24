import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { UserAvatar } from "@/components/ui/user-avatar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ArrowLeft, Users, FolderIcon, Calendar, Plus, UserMinus, Edit2, Save, X, Search } from 'lucide-react';
import { toast } from "@/hooks/use-toast";
import {
  getTeamApiV1TeamsTeamIdGet,
  listTeamMembersApiV1TeamsTeamIdMembersGet,
  getTeamSharedProjectsApiV1TeamsTeamIdSharedProjectsGet,
  getUserTeamsApiV1UserTeamsGet,
  searchUsersApiV1UsersSearchGet,
  addTeamMemberApiV1TeamsTeamIdMembersPost,
  removeTeamMemberApiV1TeamsTeamIdMembersUserIdDelete,
  updateTeamApiV1TeamsTeamIdPut
} from "@/api";
import type {
  TeamResponse,
  TeamMemberResponse,
  SharedProjectResponse,
  UserSearchResult,
  TeamMemberAdd,
  TeamUpdate
} from "@/api/types.gen";

interface TeamDetail extends TeamResponse {
  members: TeamMemberResponse[];
  shared_projects: SharedProjectResponse[];
}

export const TeamDetailPage: React.FC = () => {
  const { teamId } = useParams<{ teamId: string }>();
  const navigate = useNavigate();
  const [team, setTeam] = useState<TeamDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentUserRole, setCurrentUserRole] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedDescription, setEditedDescription] = useState('');
  const [inviteRole, setInviteRole] = useState<'member' | 'owner'>('member');
  const [showInviteForm, setShowInviteForm] = useState(false);
  const [userSearchQuery, setUserSearchQuery] = useState('');
  const [userSearchResults, setUserSearchResults] = useState<UserSearchResult[]>([]);
  const [showUserSearch, setShowUserSearch] = useState(false);
  const [selectedUser, setSelectedUser] = useState<UserSearchResult | null>(null);
  const [searchTimeoutId, setSearchTimeoutId] = useState<NodeJS.Timeout | null>(null);

  const fetchTeamDetail = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      if (!teamId) return;

      // Fetch team details, members, and shared projects in parallel
      const [teamResponse, membersResponse, sharedProjectsResponse] = await Promise.all([
        getTeamApiV1TeamsTeamIdGet({ path: { team_id: teamId } }),
        listTeamMembersApiV1TeamsTeamIdMembersGet({ path: { team_id: teamId } }),
        getTeamSharedProjectsApiV1TeamsTeamIdSharedProjectsGet({ path: { team_id: teamId } })
      ]);

      if (teamResponse.data && membersResponse.data && sharedProjectsResponse.data) {
        const teamDetail: TeamDetail = {
          ...teamResponse.data,
          members: membersResponse.data.members,
          shared_projects: sharedProjectsResponse.data.shared_projects
        };

        // Find current user's role by checking their teams from user endpoint
        try {
          const userTeamsResponse = await getUserTeamsApiV1UserTeamsGet();
          if (userTeamsResponse.data) {
            const currentTeamMembership = userTeamsResponse.data.teams.find(t => t.team_id === teamId);
            setCurrentUserRole(currentTeamMembership?.role || null);
          }
        } catch (userError) {
          console.warn('Could not fetch user role:', userError);
          setCurrentUserRole(null);
        }

        setTeam(teamDetail);
        setEditedDescription(teamResponse.data.description || '');
      }
    } catch (error) {
      console.error('Failed to fetch team details:', error);
      setError(error instanceof Error ? error.message : 'Failed to load team details');
    } finally {
      setLoading(false);
    }
  }, [teamId]);

  useEffect(() => {
    if (teamId) {
      fetchTeamDetail();
    }
  }, [teamId, fetchTeamDetail]);

  const getRoleBadgeColor = (role: string) => {
    switch (role) {
      case 'owner': return 'bg-purple-100 text-purple-800';
      case 'member': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
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


  const canManageTeam = () => {
    return currentUserRole === 'owner';
  };

  const searchUsers = async (query: string) => {
    if (query.length < 2) {
      setUserSearchResults([]);
      setShowUserSearch(false);
      return;
    }

    try {
      const response = await searchUsersApiV1UsersSearchGet({
        query: { q: query }
      });

      if (response.data) {
        setUserSearchResults(response.data.users);
        setShowUserSearch(true);
      }
    } catch (error) {
      console.error('Failed to search users:', error);
      setUserSearchResults([]);
      setShowUserSearch(false);
    }
  };

  const handleUserSearchChange = (query: string) => {
    setUserSearchQuery(query);
    setSelectedUser(null);

    // Clear existing timeout
    if (searchTimeoutId) {
      clearTimeout(searchTimeoutId);
    }

    // Set new timeout with shorter debounce (150ms)
    const newTimeoutId = setTimeout(() => {
      searchUsers(query);
    }, 150);

    setSearchTimeoutId(newTimeoutId);
  };

  const handleUserSelect = (user: UserSearchResult) => {
    setSelectedUser(user);
    setUserSearchQuery(user.full_name || user.email);
    setShowUserSearch(false);
  };

  const handleInviteMember = async () => {
    if (!selectedUser || !teamId) {
      toast({
        title: "Error",
        description: "Please select a user from the search results",
        variant: "destructive",
      });
      return;
    }

    try {
      const memberData: TeamMemberAdd = {
        email: selectedUser.email,
        role: inviteRole
      };

      await addTeamMemberApiV1TeamsTeamIdMembersPost({
        path: { team_id: teamId },
        body: memberData
      });

      toast({
        title: "Success",
        description: "Team member invited successfully",
      });

      // Reset form
      setUserSearchQuery('');
      setSelectedUser(null);
      setInviteRole('member');
      setShowInviteForm(false);
      setShowUserSearch(false);
      fetchTeamDetail(); // Refresh team data
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to invite member",
        variant: "destructive",
      });
    }
  };

  const handleRemoveMember = async (userId: string) => {
    if (!window.confirm('Are you sure you want to remove this member from the team?') || !teamId) {
      return;
    }

    try {
      await removeTeamMemberApiV1TeamsTeamIdMembersUserIdDelete({
        path: { team_id: teamId, user_id: userId }
      });

      toast({
        title: "Success",
        description: "Team member removed successfully",
      });

      fetchTeamDetail(); // Refresh team data
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to remove member",
        variant: "destructive",
      });
    }
  };

  const handleUpdateDescription = async () => {
    if (!teamId) return;

    try {
      const updateData: TeamUpdate = {
        description: editedDescription.trim()
      };

      await updateTeamApiV1TeamsTeamIdPut({
        path: { team_id: teamId },
        body: updateData
      });

      toast({
        title: "Success",
        description: "Team description updated successfully",
      });

      setIsEditing(false);
      fetchTeamDetail(); // Refresh team data
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to update description",
        variant: "destructive",
      });
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div>Loading team details...</div>
      </div>
    );
  }

  if (error || !team) {
    return (
      <div className="flex items-center justify-center py-12">
        <Card className="w-96">
          <CardContent className="pt-6">
            <div className="text-center text-red-600">{error || 'Team not found'}</div>
            <Button onClick={() => navigate('/teams')} className="w-full mt-4">
              Back to Teams
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <Button 
          variant="ghost" 
          onClick={() => navigate('/teams')}
          className="mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Teams
        </Button>
        
        <div className="flex items-center space-x-4">
          <div className="p-3 bg-blue-100 rounded-lg">
            <Users className="h-8 w-8 text-blue-600" />
          </div>
          <div className="flex-1">
            <h1 className="text-3xl font-bold text-gray-900">{team.name}</h1>
            
            {/* Description section with editing capability */}
            <div className="mt-2">
              {isEditing ? (
                <div className="space-y-2">
                  <Textarea
                    value={editedDescription}
                    onChange={(e) => setEditedDescription(e.target.value)}
                    placeholder="Enter team description..."
                    className="resize-none"
                    rows={2}
                  />
                  <div className="flex space-x-2">
                    <Button size="sm" onClick={handleUpdateDescription}>
                      <Save className="w-4 h-4 mr-1" />
                      Save
                    </Button>
                    <Button size="sm" variant="outline" onClick={() => {
                      setIsEditing(false);
                      setEditedDescription(team.description || '');
                    }}>
                      <X className="w-4 h-4 mr-1" />
                      Cancel
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <p className="text-gray-600">
                    {team.description || 'No description available'}
                  </p>
                  {canManageTeam() && (
                    <Button 
                      size="sm" 
                      variant="ghost" 
                      onClick={() => setIsEditing(true)}
                      className="p-1 h-auto"
                    >
                      <Edit2 className="w-4 h-4" />
                    </Button>
                  )}
                </div>
              )}
            </div>
            
            <div className="flex items-center space-x-4 mt-2">
              <span className="text-sm text-gray-500">
                {team.members.length} member{team.members.length !== 1 ? 's' : ''}
              </span>
              <span className="text-sm text-gray-500">
                Created {new Date(team.created_at).toLocaleDateString()}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <Tabs defaultValue="members" className="space-y-6">
        <TabsList>
          <TabsTrigger value="members">Members</TabsTrigger>
          <TabsTrigger value="projects">Shared Projects</TabsTrigger>
        </TabsList>

        <TabsContent value="members" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center">
                  <Users className="w-5 h-5 mr-2" />
                  Team Members ({team.members.length})
                </CardTitle>
                {canManageTeam() && (
                  <Button 
                    size="sm" 
                    onClick={() => setShowInviteForm(!showInviteForm)}
                    variant={showInviteForm ? "outline" : "default"}
                  >
                    <Plus className="w-4 h-4 mr-1" />
                    {showInviteForm ? 'Cancel' : 'Invite Member'}
                  </Button>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {/* Invite form */}
              {showInviteForm && canManageTeam() && (
                <div className="mb-6 p-4 border rounded-lg bg-gray-50">
                  <h4 className="font-medium mb-3">Invite New Member</h4>
                  
                  {/* User search */}
                  <div className="space-y-3">
                    <div className="relative">
                      <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                      <Input
                        placeholder="Search users by name or email..."
                        value={userSearchQuery}
                        onChange={(e) => handleUserSearchChange(e.target.value)}
                        className="pl-10"
                      />
                      
                      {/* Search results dropdown */}
                      {showUserSearch && userSearchResults.length > 0 && (
                        <div className="absolute z-10 w-full mt-1 bg-white border rounded-md shadow-lg max-h-60 overflow-auto">
                          {userSearchResults.map((user) => (
                            <div
                              key={user.id}
                              className="px-3 py-2 hover:bg-gray-100 cursor-pointer border-b last:border-b-0"
                              onClick={() => handleUserSelect(user)}
                            >
                              <div className="font-medium">{user.full_name || user.email}</div>
                              <div className="text-sm text-gray-500">{user.email}</div>
                            </div>
                          ))}
                        </div>
                      )}
                      
                      {/* No results */}
                      {showUserSearch && userSearchQuery.length >= 2 && userSearchResults.length === 0 && (
                        <div className="absolute z-10 w-full mt-1 bg-white border rounded-md shadow-lg p-3 text-center text-gray-500">
                          No users found
                        </div>
                      )}
                    </div>
                    
                    {/* Role selection */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Role
                      </label>
                      <Select value={inviteRole} onValueChange={(value: 'member' | 'owner') => setInviteRole(value)}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="member">Member - Can view team and projects</SelectItem>
                          <SelectItem value="owner">Owner - Can manage team and members</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    {/* Action buttons */}
                    <div className="flex space-x-2 pt-2">
                      <Button onClick={handleInviteMember} className="flex-1">
                        Send Invite
                      </Button>
                      <Button 
                        variant="outline" 
                        onClick={() => {
                          setShowInviteForm(false);
                          setUserSearchQuery('');
                          setSelectedUser(null);
                          setShowUserSearch(false);
                          setInviteRole('member');
                        }}
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                </div>
              )}
              
              <div className="space-y-4">
                {team.members.map((member) => (
                  <div key={member.user_id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <UserAvatar
                        userId={member.user_id}
                        user={{
                          email: member.email,
                          full_name: member.full_name
                        }}
                        size="lg"
                      />
                      <div>
                        <h4 className="font-medium text-gray-900">
                          {member.full_name || member.email}
                        </h4>
                        <p className="text-sm text-gray-500">{member.email}</p>
                        <p className="text-xs text-gray-400">
                          Joined {new Date(member.joined_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge className={getRoleBadgeColor(member.role)}>
                        {member.role}
                      </Badge>
                      {canManageTeam() && member.role !== 'owner' && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleRemoveMember(member.user_id)}
                          className="text-red-600 hover:text-red-700 hover:bg-red-50"
                        >
                          <UserMinus className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="projects" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <FolderIcon className="w-5 h-5 mr-2" />
                Shared Projects ({team.shared_projects.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              {team.shared_projects.length === 0 ? (
                <div className="text-center py-8">
                  <FolderIcon className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No shared projects</h3>
                  <p className="text-gray-500">
                    This team doesn't have any shared projects yet.
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {team.shared_projects.map((project) => (
                    <div key={project.id} className="p-4 border rounded-lg hover:shadow-sm transition-shadow">
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-medium text-gray-900">{project.name}</h4>
                          {project.description && (
                            <p className="text-sm text-gray-600 mt-1">{project.description}</p>
                          )}
                          <div className="flex items-center space-x-4 mt-2">
                            <Badge className={getAccessBadgeColor(project.access_mode)}>
                              {project.access_mode}
                            </Badge>
                            <span className="text-xs text-gray-400 flex items-center">
                              <Calendar className="w-3 h-3 mr-1" />
                              Updated {new Date(project.updated_at).toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => navigate(`/projects/${project.id}`)}
                        >
                          View Project
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};