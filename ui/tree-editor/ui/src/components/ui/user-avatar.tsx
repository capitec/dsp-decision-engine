import React, { useState, useEffect } from 'react';
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { getOtherUserProfileApiV1UserUserIdProfileGet } from "@/api";

interface UserAvatarProps {
  userId?: string;
  user?: {
    email: string;
    full_name?: string | null;
    profile_picture_url?: string | null;
  };
  size?: 'sm' | 'default' | 'lg';
  className?: string;
}

interface UserProfileData {
  id: string;
  email: string;
  full_name: string | null;
  profile_picture_url: string | null;
}

const sizeClasses = {
  sm: 'h-6 w-6',
  default: 'h-8 w-8',
  lg: 'h-10 w-10'
};

// Cache for user profiles to avoid refetching
const userCache = new Map<string, UserProfileData>();

export const UserAvatar: React.FC<UserAvatarProps> = ({
  userId,
  user,
  size = 'default',
  className = ''
}) => {
  const [profileData, setProfileData] = useState<UserProfileData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // If user data is provided directly, use it
    if (user) {
      setProfileData({
        id: userId || '',
        email: user.email,
        full_name: user.full_name || null,
        profile_picture_url: user.profile_picture_url || null
      });
      return;
    }

    // If userId is provided but no user data, fetch it
    if (userId && !userCache.has(userId)) {
      const fetchUserProfile = async () => {
        setLoading(true);
        try {
          const response = await getOtherUserProfileApiV1UserUserIdProfileGet({
            path: { user_id: userId }
          });

          if (response.data) {
            const profile: UserProfileData = {
              id: response.data.id,
              email: response.data.email,
              full_name: response.data.full_name,
              profile_picture_url: response.data.profile_picture_url || null
            };
            userCache.set(userId, profile);
            setProfileData(profile);
          }
        } catch (error) {
          console.error('Failed to fetch user profile:', error);
          // Set basic data to prevent infinite loading
          const fallbackProfile: UserProfileData = {
            id: userId,
            email: 'Unknown',
            full_name: null,
            profile_picture_url: null
          };
          setProfileData(fallbackProfile);
        } finally {
          setLoading(false);
        }
      };

      fetchUserProfile();
    } else if (userId && userCache.has(userId)) {
      // Use cached data
      setProfileData(userCache.get(userId)!);
    }
  }, [userId, user]);

  const getUserInitials = () => {
    if (profileData?.full_name) {
      return profileData.full_name.split(' ').map(n => n[0]).join('').toUpperCase();
    }
    if (profileData?.email) {
      return profileData.email[0].toUpperCase();
    }
    return 'U';
  };

  if (loading) {
    return (
      <Avatar className={`${sizeClasses[size]} ${className}`}>
        <AvatarFallback>...</AvatarFallback>
      </Avatar>
    );
  }

  return (
    <Avatar className={`${sizeClasses[size]} ${className}`}>
      <AvatarImage 
        src={profileData?.profile_picture_url || undefined} 
        alt={profileData?.full_name || profileData?.email || 'User'} 
      />
      <AvatarFallback>{getUserInitials()}</AvatarFallback>
    </Avatar>
  );
};