import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Users, ChevronRight, Plus } from 'lucide-react';
import { toast } from "@/hooks/use-toast";
import { getUserTeamsApiV1UserTeamsGet, createTeamApiV1TeamsPost } from "@/api";
import type { TeamMembershipResponse, TeamCreate } from "@/api/types.gen";

export const TeamsPage: React.FC = () => {
  const navigate = useNavigate();
  const [teams, setTeams] = useState<TeamMembershipResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newTeamName, setNewTeamName] = useState('');
  const [newTeamDescription, setNewTeamDescription] = useState('');

  useEffect(() => {
    fetchTeams();
  }, []);

  const fetchTeams = async () => {
    try {
      setLoading(true);
      const response = await getUserTeamsApiV1UserTeamsGet();
      if (response.data) {
        setTeams(response.data.teams);
      }
    } catch (error) {
      console.error('Failed to fetch teams:', error);
      setError('Failed to load teams');
    } finally {
      setLoading(false);
    }
  };

  const getRoleBadgeColor = (role: string) => {
    switch (role) {
      case 'owner': return 'bg-purple-100 text-purple-800';
      case 'member': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const handleTeamClick = (teamId: string) => {
    navigate(`/teams/${teamId}`);
  };

  const handleCreateTeam = async () => {
    if (!newTeamName.trim()) {
      toast({
        title: "Error",
        description: "Please enter a team name",
        variant: "destructive",
      });
      return;
    }

    try {
      const teamData: TeamCreate = {
        name: newTeamName.trim(),
        description: newTeamDescription.trim() || undefined
      };

      await createTeamApiV1TeamsPost({
        body: teamData
      });

      toast({
        title: "Success",
        description: "Team created successfully",
      });

      setNewTeamName('');
      setNewTeamDescription('');
      setShowCreateForm(false);
      fetchTeams(); // Refresh teams list
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to create team",
        variant: "destructive",
      });
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div>Loading teams...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-12">
        <Card className="w-96">
          <CardContent className="pt-6">
            <div className="text-center text-red-600">{error}</div>
            <Button onClick={fetchTeams} className="w-full mt-4">
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Teams</h1>
          <p className="text-gray-600 mt-2">
            Teams you belong to and collaborate with
          </p>
        </div>
        <Button 
          onClick={() => setShowCreateForm(!showCreateForm)}
          variant={showCreateForm ? "outline" : "default"}
        >
          <Plus className="w-4 h-4 mr-1" />
          {showCreateForm ? 'Cancel' : 'Create Team'}
        </Button>
      </div>

      {/* Create team form */}
      {showCreateForm && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Create New Team</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Team Name *
                </label>
                <Input
                  placeholder="Enter team name"
                  value={newTeamName}
                  onChange={(e) => setNewTeamName(e.target.value)}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <Textarea
                  placeholder="Enter team description (optional)"
                  value={newTeamDescription}
                  onChange={(e) => setNewTeamDescription(e.target.value)}
                  rows={3}
                />
              </div>
              <div className="flex space-x-2">
                <Button onClick={handleCreateTeam}>
                  Create Team
                </Button>
                <Button 
                  variant="outline" 
                  onClick={() => {
                    setShowCreateForm(false);
                    setNewTeamName('');
                    setNewTeamDescription('');
                  }}
                >
                  Cancel
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {teams.length === 0 ? (
        <Card>
          <CardContent className="pt-6 text-center py-12">
            <Users className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No teams yet</h3>
            <p className="text-gray-500 mb-4">
              You haven't been added to any teams yet. Contact your administrator to join teams.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {teams.map((team) => (
            <Card 
              key={team.id} 
              className="hover:shadow-md transition-shadow cursor-pointer"
              onClick={() => handleTeamClick(team.team_id)}
            >
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <Users className="h-6 w-6 text-blue-600" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900">
                        {team.team_name}
                      </h3>
                      {team.team_description && (
                        <p className="text-gray-600 mt-1">
                          {team.team_description}
                        </p>
                      )}
                      <div className="flex items-center space-x-4 mt-2">
                        <Badge className={getRoleBadgeColor(team.role)}>
                          {team.role}
                        </Badge>
                        <span className="text-sm text-gray-500">
                          Joined {new Date(team.joined_at).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  </div>
                  <ChevronRight className="h-5 w-5 text-gray-400" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};