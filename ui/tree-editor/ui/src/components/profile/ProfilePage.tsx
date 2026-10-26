import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { toast } from "@/hooks/use-toast";
import {
  getUserProfileApiV1UserProfileGet,
  updateUserProfileApiV1UserProfilePut,
  uploadProfilePictureApiV1UserProfilePicturePost,
  deleteProfilePictureApiV1UserProfilePictureDelete
} from "@/api";
import type { UserProfileResponse } from "@/api/types.gen";

export const ProfilePage: React.FC = () => {
  const [profile, setProfile] = useState<UserProfileResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Form state
  const [fullName, setFullName] = useState('');

  useEffect(() => {
    fetchProfileData();
  }, []);

  const fetchProfileData = async () => {
    try {
      setLoading(true);

      const response = await getUserProfileApiV1UserProfileGet();

      if (response.data) {
        setProfile(response.data);
        // Initialize form with current values
        setFullName(response.data.full_name || '');
      }

    } catch (error) {
      console.error('Failed to fetch profile data:', error);
      setError('Failed to load profile information');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveProfile = async () => {
    if (!profile) return;
    
    try {
      setSaving(true);
      
      const updateData: { full_name?: string } = {};
      
      // Only include changed fields
      if (fullName !== (profile.full_name || '')) {
        updateData.full_name = fullName;
      }
      
      // If nothing changed, don't make the request
      if (Object.keys(updateData).length === 0) {
        toast({
          title: "No changes",
          description: "No changes were made to your profile.",
        });
        return;
      }
      
      const response = await updateUserProfileApiV1UserProfilePut({
        body: updateData
      });

      if (response.data) {
        setProfile(response.data);
      }
      
      toast({
        title: "Profile updated",
        description: "Your profile has been successfully updated.",
      });
      
    } catch (error: any) {
      console.error('Failed to update profile:', error);
      toast({
        title: "Update failed",
        description: error.message || "Failed to update your profile. Please try again.",
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  };

  const handleProfilePictureUpload = async (file: File) => {
    try {
      setUploading(true);

      const response = await uploadProfilePictureApiV1UserProfilePicturePost({
        body: { file }
      });

      if (response.data) {
        setProfile(response.data);
      }

      toast({
        title: "Profile picture updated",
        description: "Your profile picture has been successfully updated.",
      });

    } catch (error: any) {
      console.error('Failed to upload profile picture:', error);
      toast({
        title: "Upload failed",
        description: error.message || "Failed to upload profile picture. Please try again.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteProfilePicture = async () => {
    try {
      setUploading(true);

      const response = await deleteProfilePictureApiV1UserProfilePictureDelete();

      if (response.data) {
        setProfile(response.data);
      }

      toast({
        title: "Profile picture removed",
        description: "Your profile picture has been removed.",
      });

    } catch (error: any) {
      console.error('Failed to delete profile picture:', error);
      toast({
        title: "Delete failed",
        description: error.message || "Failed to delete profile picture. Please try again.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };


  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div>Loading profile...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="w-96">
          <CardContent className="pt-6">
            <div className="text-center text-red-600">{error}</div>
            <Button onClick={fetchProfileData} className="w-full mt-4">
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div>Profile not found</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">Profile Settings</h1>
        
        {/* Profile Information */}
        <Card>
            <CardHeader>
              <CardTitle>Personal Information</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Profile Picture Section */}
              <div className="text-center">
                <div className="mb-4">
                  {profile.profile_picture_url ? (
                    <img 
                      src={profile.profile_picture_url} 
                      alt="Profile" 
                      className="w-24 h-24 rounded-full mx-auto object-cover border-2 border-gray-200"
                    />
                  ) : (
                    <div className="w-24 h-24 rounded-full mx-auto bg-gray-200 flex items-center justify-center">
                      <span className="text-gray-500 text-xl">
                        {profile.full_name?.[0]?.toUpperCase() || profile.email[0].toUpperCase()}
                      </span>
                    </div>
                  )}
                </div>
                
                <div className="flex gap-2 justify-center">
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        handleProfilePictureUpload(file);
                      }
                    }}
                    className="hidden"
                    id="profile-picture-upload"
                  />
                  <Button 
                    onClick={() => document.getElementById('profile-picture-upload')?.click()}
                    disabled={uploading}
                    variant="outline"
                    size="sm"
                  >
                    {uploading ? 'Uploading...' : 'Upload Picture'}
                  </Button>
                  
                  {profile.profile_picture_url && (
                    <Button 
                      onClick={handleDeleteProfilePicture}
                      disabled={uploading}
                      variant="outline"
                      size="sm"
                    >
                      Remove
                    </Button>
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Max 2MB. JPEG, PNG, or WebP formats.
                </p>
              </div>
              
              <Separator />
              
              <div>
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  value={profile.email}
                  disabled
                  className="bg-gray-50"
                />
                <p className="text-sm text-gray-500 mt-1">
                  Email cannot be changed
                </p>
              </div>
              
              <div>
                <Label htmlFor="fullName">Full Name</Label>
                <Input
                  id="fullName"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Enter your full name"
                />
              </div>
              
              <div>
                <Label>Account Status</Label>
                <div className="mt-1">
                  <Badge className={profile.is_active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                    {profile.is_active ? 'Active' : 'Inactive'}
                  </Badge>
                </div>
              </div>
              
              <div>
                <Label>Member Since</Label>
                <p className="text-sm text-gray-600 mt-1">
                  {new Date(profile.created_at).toLocaleDateString()}
                </p>
              </div>
              
              <Separator />
              
              <Button 
                onClick={handleSaveProfile} 
                disabled={saving}
                className="w-full"
              >
                {saving ? 'Saving...' : 'Save Changes'}
              </Button>
            </CardContent>
          </Card>
      </div>
    </div>
  );
};