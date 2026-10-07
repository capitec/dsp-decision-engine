import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import { SessionRefreshDialog } from '../components/auth/SessionRefreshDialog';
import { isAuthenticated, getSessionStatus, isOnPublicPage } from '../lib/auth';
import { toast } from '@/hooks/use-toast';

// Calculate a reasonable interval for checking session status
const calculateSessionCheckInterval = (): number => {
  // In debug mode (rolling=15s, threshold=10s): check every 5s  
  // In production (rolling=3600s, threshold=900s): check every ~450s (7.5 min)
  // Formula: min(30000, max(5000, (rolling_duration - renewal_threshold) / 2))
  
  // These would ideally come from backend config, but for now use reasonable defaults
  const isDebugMode = window.location.hostname === 'localhost';
  const rollingDuration = isDebugMode ? 15 : 3600; // seconds
  const renewalThreshold = isDebugMode ? 10 : 900;  // seconds
  
  // Check halfway between renewal point and expiry, with reasonable bounds
  const calculatedInterval = Math.max(5, (rollingDuration - renewalThreshold) / 2) * 1000;
  const maxInterval = 30000; // Never more than 30 seconds
  
  return Math.min(maxInterval, calculatedInterval);
};

interface SessionContextType {
  showSessionRefresh: (message?: string) => void;
  hideSessionRefresh: () => void;
  isSessionDialogOpen: boolean;
  checkSessionValidity: () => boolean;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

export const useSession = () => {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
};

interface SessionProviderProps {
  children: React.ReactNode;
}

export const SessionProvider: React.FC<SessionProviderProps> = ({ children }) => {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [dialogMessage, setDialogMessage] = useState<string>();
  const pendingRequestsRef = useRef<Array<{ resolve: () => void; reject: (error: Error) => void }>>([]);
  const sessionCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastRefreshAttemptRef = useRef<number>(0);

  // Proactive session monitoring
  useEffect(() => {
    // Don't start monitoring on public pages (login, unauthorized)
    if (isOnPublicPage()) {
      console.log('SessionContext: On public page, skipping session monitoring');
      return;
    }

    // Initialize with current session expiry
    const initialSessionStatus = getSessionStatus();
    let lastKnownExpiry = initialSessionStatus ? parseInt(initialSessionStatus, 10) : 0;
    const isDebugMode = window.location.hostname === 'localhost';

    const startSessionMonitoring = () => {

      // Check session every 30 seconds
      sessionCheckIntervalRef.current = setInterval(() => {
        // Get current session status before checking if expired
        const currentSessionStatus = getSessionStatus();
        const currentExpiry = currentSessionStatus ? parseInt(currentSessionStatus, 10) : 0;
        
        // If expiry time has changed (session was renewed), update our tracking
        if (currentExpiry !== lastKnownExpiry) {
          console.log(`Session expiry updated: ${lastKnownExpiry} -> ${currentExpiry} (raw cookie: "${currentSessionStatus}")`);
          lastKnownExpiry = currentExpiry;
          
          // If session was renewed and dialog is open, we can close it
          if (isAuthenticated() && isDialogOpen) {
            console.log('Session was renewed, closing dialog');
            setIsDialogOpen(false);
            return;
          }
        }
        
        // Show dialog if session is actually expired and user is on a protected page
        // But give a grace period for automatic renewal to work first
        if (!isAuthenticated()) {
          // Only show dialog on pages that require authentication
          // Login page and unauthorized page are public and should not trigger the dialog
          if (!isOnPublicPage() && !isDialogOpen) {
            const currentTime = Math.floor(Date.now() / 1000);
            const timeSinceExpiry = currentTime - currentExpiry;

            // Only show dialog if session has been expired for more than the renewal threshold
            // This gives automatic renewal a chance to work
            const renewalThreshold = isDebugMode ? 10 : 900; // Match backend settings
            if (timeSinceExpiry > renewalThreshold) {
              console.log(`Session expired ${timeSinceExpiry}s ago, showing dialog (expiry: ${currentExpiry}, now: ${currentTime})`);
              showSessionRefresh("Your session has expired. Please refresh your authentication to continue working.");
            }
          }
        }
        
        // Auto-close dialog if session gets renewed
        else if (isDialogOpen) {
          console.log('Session was renewed while dialog was open, closing dialog');
          setIsDialogOpen(false);
        }
      }, calculateSessionCheckInterval()); // Check at a reasonable interval based on session settings
    };

    // Start monitoring if already authenticated
    if (isAuthenticated()) {
      startSessionMonitoring();
    }

    // Listen for authentication events
    const handleLogin = () => {
      startSessionMonitoring();
    };

    const handleSessionRefresh = () => {
      // Session was refreshed, restart monitoring
      startSessionMonitoring();
    };

    const handleLogout = () => {
      if (sessionCheckIntervalRef.current) {
        clearInterval(sessionCheckIntervalRef.current);
        sessionCheckIntervalRef.current = null;
      }
    };

    window.addEventListener('auth-login', handleLogin);
    window.addEventListener('session-refreshed', handleSessionRefresh);
    window.addEventListener('auth-logout', handleLogout);

    return () => {
      if (sessionCheckIntervalRef.current) {
        clearInterval(sessionCheckIntervalRef.current);
      }
      window.removeEventListener('auth-login', handleLogin);
      window.removeEventListener('session-refreshed', handleSessionRefresh);
      window.removeEventListener('auth-logout', handleLogout);
    };
  }, [isDialogOpen]);

  const showSessionRefresh = useCallback((message?: string) => {
    setDialogMessage(message);
    setIsDialogOpen(true);
  }, []);

  const hideSessionRefresh = useCallback(() => {
    setIsDialogOpen(false);
    setDialogMessage(undefined);
  }, []);

  const checkSessionValidity = useCallback((): boolean => {
    return isAuthenticated();
  }, []);

  const handleSessionRestored = useCallback(() => {
    console.log('SessionContext: Session restored, resolving pending requests');
    
    // Resolve all pending requests
    pendingRequestsRef.current.forEach(({ resolve }) => {
      console.log('SessionContext: Resolving pending request');
      resolve();
    });
    pendingRequestsRef.current = [];
    
    // Hide dialog
    hideSessionRefresh();
    
    // Show success toast
    toast({
      title: "Session Restored",
      description: "Your authentication has been refreshed successfully.",
    });

    // Trigger session-refreshed event instead of auth-login to avoid navigation
    window.dispatchEvent(new CustomEvent('session-refreshed'));
  }, [hideSessionRefresh]);

  const handleForceLogout = useCallback(() => {
    // Reject all pending requests
    pendingRequestsRef.current.forEach(({ reject }) => 
      reject(new Error('User chose to log out'))
    );
    pendingRequestsRef.current = [];
    
    // Hide dialog
    hideSessionRefresh();
    
    // Perform logout
    window.location.href = '/api/auth/logout';
  }, [hideSessionRefresh]);

  // Method to wait for session refresh (used by fetch interceptor)
  const waitForSessionRefresh = useCallback((): Promise<void> => {
    return new Promise((resolve, reject) => {
      console.log('SessionContext: waitForSessionRefresh called - server rejected request with 401');

      // Don't show dialog on public pages (login, unauthorized)
      if (isOnPublicPage()) {
        console.log('SessionContext: On public page, rejecting without showing dialog');
        reject(new Error('Authentication required'));
        return;
      }

      const now = Date.now();
      const timeSinceLastAttempt = now - lastRefreshAttemptRef.current;

      // If we just attempted a refresh within the last 2 seconds, don't spam
      if (timeSinceLastAttempt < 2000 && isDialogOpen) {
        console.log('SessionContext: Recent refresh attempt in progress, adding to queue');
        pendingRequestsRef.current.push({ resolve, reject });
        return;
      }

      // Don't trust client-side authentication state when server says 401
      // Always force session refresh dialog when this is called
      console.log('SessionContext: Adding to pending requests (ignoring client-side auth state)');
      lastRefreshAttemptRef.current = now;

      // Add to pending requests
      pendingRequestsRef.current.push({ resolve, reject });

      // Show dialog if not already shown
      if (!isDialogOpen) {
        console.log('SessionContext: Showing session refresh dialog');
        showSessionRefresh("Your session has expired. Please refresh your authentication to continue the operation.");
      }
    });
  }, [isDialogOpen, showSessionRefresh]);

  // Expose waitForSessionRefresh globally for fetch interceptor
  useEffect(() => {
    (window as any).__waitForSessionRefresh = waitForSessionRefresh;
    return () => {
      delete (window as any).__waitForSessionRefresh;
    };
  }, [waitForSessionRefresh]);

  const contextValue: SessionContextType = {
    showSessionRefresh,
    hideSessionRefresh,
    isSessionDialogOpen: isDialogOpen,
    checkSessionValidity,
  };

  return (
    <SessionContext.Provider value={contextValue}>
      {children}
      <SessionRefreshDialog
        open={isDialogOpen}
        onClose={hideSessionRefresh}
        onSessionRestored={handleSessionRestored}
        onForceLogout={handleForceLogout}
        message={dialogMessage}
      />
    </SessionContext.Provider>
  );
};