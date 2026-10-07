import React, { useEffect, useState, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import './App.css';
import { Toaster } from "@/components/ui/toaster";
import { SessionProvider } from "../../contexts/SessionContext";
import { FolderProvider } from "../../contexts/FolderContext";
import { AuthSuccessHandler } from "../auth/AuthSuccessHandler";
import { isAuthenticated } from "../../lib/auth";
import { getUserProfileApiV1UserProfileGet } from "@/api";
const MainLayout = React.lazy(() => import("../layout/MainLayout").then(module => ({ default: module.MainLayout })));

// Lazy load components for code splitting
const LoginPage = React.lazy(() => import("../auth/LoginPage").then(module => ({ default: module.LoginPage })));
const UnauthorizedPage = React.lazy(() => import("../auth/UnauthorizedPage").then(module => ({ default: module.UnauthorizedPage })));
const OverviewPage = React.lazy(() => import("../overview/OverviewPage").then(module => ({ default: module.OverviewPage })));
const ProjectListPage = React.lazy(() => import("../projects/ProjectListPage").then(module => ({ default: module.ProjectListPage })));
const TeamsPage = React.lazy(() => import("../teams/TeamsPage").then(module => ({ default: module.TeamsPage })));
const TeamDetailPage = React.lazy(() => import("../teams/TeamDetailPage").then(module => ({ default: module.TeamDetailPage })));
const ProfilePage = React.lazy(() => import("../profile/ProfilePage").then(module => ({ default: module.ProfilePage })));
const ApiTokensPage = React.lazy(() => import("../profile/ApiTokensPage").then(module => ({ default: module.ApiTokensPage })));
const ProjectEditorWrapper = React.lazy(() => import("../project-editor/ProjectEditorWrapper"));
const FlatRuleEditorPage = React.lazy(() => import("../flat-rules/FlatRuleEditorPage").then(module => ({ default: module.FlatRuleEditorPage })));
const DatasetsPage = React.lazy(() => import("../datasets/DatasetsPage").then(module => ({ default: module.DatasetsPage })));

// Protected Route component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  if (isAuthenticated()) {
    return <>{children}</>;
  } else {
    // Redirect to logout endpoint to clear session properly
    window.location.href = '/api/auth/logout';
    return null;
  }
};

// Auth listener component to handle login/logout events
const AuthListener: React.FC = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const handleLogin = () => {
      // Only navigate on fresh login, not session refresh
      navigate('/');
    };

    const handleLogout = () => {
      // Redirect to logout endpoint to clear session properly
      window.location.href = '/api/auth/logout';
    };

    const handleSessionRefresh = () => {
      // Session was refreshed - do NOT navigate, just log
      console.log('AuthListener: Session refreshed, staying on current page');
    };

    window.addEventListener('auth-login', handleLogin);
    window.addEventListener('auth-logout', handleLogout);
    window.addEventListener('session-refreshed', handleSessionRefresh);

    return () => {
      window.removeEventListener('auth-login', handleLogin);
      window.removeEventListener('auth-logout', handleLogout);
      window.removeEventListener('session-refreshed', handleSessionRefresh);
    };
  }, [navigate]);

  return null;
};

import type { UserProfileResponse } from "@/api/types.gen";

// Layout wrapper for protected routes
const LayoutWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentUser, setCurrentUser] = useState<UserProfileResponse | null>(null);

  useEffect(() => {
    const fetchUserProfile = async () => {
      try {
        const response = await getUserProfileApiV1UserProfileGet();
        if (response.data) {
          setCurrentUser(response.data);
        }
      } catch (error) {
        console.error('Failed to fetch user profile:', error);
      }
    };

    if (isAuthenticated()) {
      fetchUserProfile();
    }
  }, []);

  const mappedUser = currentUser ? {
    id: currentUser.id,
    email: currentUser.email,
    full_name: currentUser.full_name || undefined,
    profile_picture_url: currentUser.profile_picture_url || undefined
  } : undefined;

  return (
    <MainLayout currentUser={mappedUser}>
      <AuthSuccessHandler />
      {children}
    </MainLayout>
  );
};

function App() {
  return (
    <Router>
      <SessionProvider>
        <FolderProvider>
          <div className="App">
            <AuthListener />
            <Suspense fallback={
              <div className="min-h-screen flex items-center justify-center">
                <div className="text-lg">Loading...</div>
              </div>
            }>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route path="/unauthorized" element={<UnauthorizedPage />} />
              <Route 
                path="/" 
                element={
                  <ProtectedRoute>
                    <LayoutWrapper>
                      <OverviewPage />
                    </LayoutWrapper>
                  </ProtectedRoute>
                } 
              />
              <Route 
                path="/projects" 
                element={
                  <ProtectedRoute>
                    <LayoutWrapper>
                      <ProjectListPage />
                    </LayoutWrapper>
                  </ProtectedRoute>
                } 
              />
              <Route 
                path="/projects/:projectId" 
                element={
                  <ProtectedRoute>
                    <ProjectEditorWrapper />
                  </ProtectedRoute>
                } 
              />
              <Route 
                path="/teams" 
                element={
                  <ProtectedRoute>
                    <LayoutWrapper>
                      <TeamsPage />
                    </LayoutWrapper>
                  </ProtectedRoute>
                } 
              />
              <Route
                path="/teams/:teamId"
                element={
                  <ProtectedRoute>
                    <LayoutWrapper>
                      <TeamDetailPage />
                    </LayoutWrapper>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/profile" 
                element={
                  <ProtectedRoute>
                    <LayoutWrapper>
                      <ProfilePage />
                    </LayoutWrapper>
                  </ProtectedRoute>
                } 
              />
              <Route 
                path="/api-tokens" 
                element={
                  <ProtectedRoute>
                    <LayoutWrapper>
                      <ApiTokensPage />
                    </LayoutWrapper>
                  </ProtectedRoute>
                } 
              />
              <Route
                path="/flat-rules/:projectId"
                element={
                  <ProtectedRoute>
                    <FlatRuleEditorPage />
                  </ProtectedRoute>
                }
              />
              <Route 
                path="/datasets" 
                element={
                  <ProtectedRoute>
                    <LayoutWrapper>
                      <DatasetsPage />
                    </LayoutWrapper>
                  </ProtectedRoute>
                } 
              />
              {/* Default redirect */}
              <Route path="*" element={<Navigate to="/" />} />
            </Routes>
          </Suspense>
          <Toaster />
          </div>
        </FolderProvider>
      </SessionProvider>
    </Router>
  );
}

export default App;


