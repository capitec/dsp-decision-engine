import { toast } from "@/hooks/use-toast";

/**
 * Authentication utilities for session management and token refresh.
 *
 * NOTE: The generated API client (from @hey-api/openapi-ts) uses the same
 * auth logic via interceptors in `lib/auth-interceptors.ts`. Both fetchJSON
 * and the generated SDK functions use identical session refresh behavior.
 */

// Get session status cookie value (contains expiry timestamp)
export const getSessionStatus = (): string | null => {
  const cookies = document.cookie.split(';');
  const statusCookie = cookies.find(cookie => 
    cookie.trim().startsWith('session_status=')
  );
  return statusCookie ? statusCookie.split('=')[1] : null;
};

// Check if user is authenticated based on session status cookie expiry
export const isAuthenticated = (): boolean => {
  const statusValue = getSessionStatus();
  if (!statusValue) {
    return false;
  }

  // Parse the expiry timestamp and compare with current time
  const expiryTimestamp = parseInt(statusValue, 10);
  if (isNaN(expiryTimestamp)) {
    return false;
  }

  const currentTimestamp = Math.floor(Date.now() / 1000);
  return currentTimestamp < expiryTimestamp;
};

// Check if the current page is a public page that doesn't require authentication
export const isOnPublicPage = (pathname: string = window.location.pathname): boolean => {
  return pathname === '/login' ||
         pathname.startsWith('/login/') ||
         pathname === '/unauthorized' ||
         pathname.startsWith('/unauthorized/');
};

// Clear session and redirect to login
export const logout = async (message?: string) => {
  // Store message in sessionStorage to show after redirect
  if (message) {
    sessionStorage.setItem('logout_message', message);
  }
  
  // Redirect to server-side logout endpoint
  // This will clear all cookies (including httponly ones) and redirect to login
  window.location.href = '/api/auth/logout';
};

// Custom fetch wrapper with auth interceptors
export const authenticatedFetch = async (
  url: string, 
  options: RequestInit = {},
  retryCount = 0
): Promise<Response> => {
  const response = await fetch(url, {
    ...options,
    credentials: 'include' // Always include cookies
  });
  
  // If we get a 401, check if we have session refresh capability
  if (response.status === 401) {
    // Prevent infinite retry loops
    if (retryCount >= 2) {
      console.error('authenticatedFetch: Maximum retries exceeded for', url);
      throw new Error('Authentication failed - maximum retries exceeded');
    }
    
    const waitForSessionRefresh = (window as any).__waitForSessionRefresh;
    
    if (waitForSessionRefresh && typeof waitForSessionRefresh === 'function') {
      // Try to refresh the session using the popup dialog
      try {
        console.log(`authenticatedFetch: Attempting session refresh (retry ${retryCount + 1}) for`, url);
        await waitForSessionRefresh();
        
        // Give a moment for cookies to propagate and verify session is actually restored
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Double-check that session is actually restored before retrying
        if (!isAuthenticated()) {
          throw new Error('Session refresh failed - authentication not restored');
        }
        
        // Session was refreshed, retry the original request with incremented count
        console.log(`authenticatedFetch: Session refreshed, retrying request for`, url);
        return authenticatedFetch(url, options, retryCount + 1);
      } catch (refreshError) {
        // User chose to logout or refresh failed
        console.error('Session refresh failed:', refreshError);
        throw new Error('Authentication required');
      }
    } else {
      // Fallback to old behavior if session context is not available
      logout("Your session has expired. Please sign in again.");
      throw new Error('Authentication required');
    }
  }
  
  return response;
};

// Convenience wrapper for JSON requests
export const fetchJSON = async <T = any>(
  url: string, 
  options: RequestInit = {}
): Promise<T> => {
  const response = await authenticatedFetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Request failed' }));
    
    // Provide more specific error messages but preserve full error data
    if (response.status === 401) {
      throw new Error('Not Authorized - session refresh failed or was cancelled');
    } else if (response.status === 403) {
      throw new Error('Access Forbidden - insufficient permissions');
    } else {
      // Create error that preserves both message and full error data
      const error = new Error(
        typeof errorData === 'string' ? errorData : 
        errorData.message || errorData.detail || `Request failed with status ${response.status}`
      );
      // Attach the full error data to the error object for detailed error handling
      (error as any).detail = errorData;
      (error as any).status = response.status;
      throw error;
    }
  }
  
  return response.json();
};