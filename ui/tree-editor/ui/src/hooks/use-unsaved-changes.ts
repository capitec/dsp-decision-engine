import { useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

/**
 * Custom hook to warn users about unsaved changes when they try to navigate away
 * or close the browser tab/window.
 */
export const useUnsavedChanges = (isDirty: boolean, message = "You have unsaved changes. Are you sure you want to leave?") => {
  const navigate = useNavigate();
  const location = useLocation();

  // Handle browser events (refresh, close tab, back/forward)
  useEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      if (isDirty) {
        // Standard way to show browser warning
        event.preventDefault();
        event.returnValue = message;
        return message;
      }
    };

    // Add event listener for browser navigation/refresh/close
    if (isDirty) {
      window.addEventListener('beforeunload', handleBeforeUnload);
    }

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [isDirty, message]);

  // Handle React Router navigation
  useEffect(() => {
    if (isDirty) {
      // Store current path when becoming dirty
      const currentPath = location.pathname;
      
      const handlePopState = (event: PopStateEvent) => {
        if (isDirty && !window.confirm(message)) {
          // If user cancels, push current path back to history
          window.history.pushState(null, '', currentPath);
          event.preventDefault();
        }
      };

      window.addEventListener('popstate', handlePopState);

      return () => {
        window.removeEventListener('popstate', handlePopState);
      };
    }
  }, [isDirty, message, location.pathname]);

  // Return a function to safely navigate (with confirmation)
  const navigateWithConfirmation = useCallback((to: string) => {
    if (isDirty) {
      if (window.confirm(message)) {
        navigate(to);
      }
    } else {
      navigate(to);
    }
  }, [isDirty, message, navigate]);

  return { navigateWithConfirmation };
};