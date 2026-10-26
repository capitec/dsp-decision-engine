/**
 * Authentication interceptors for the API client
 *
 * These interceptors handle:
 * - Adding credentials to all requests
 * - Automatic session refresh on 401 responses
 */

import { isAuthenticated, logout } from './auth';

let refreshInProgress = false;
let refreshPromise: Promise<void> | null = null;

// A sent request's body is consumed, so a request retried after a session refresh is sent from an unsent copy
const unsentCopies = new WeakMap<Request, Request>();
const retry = (request: Request) => fetch(unsentCopies.get(request) ?? request);

/**
 * Response interceptor that handles 401 errors with automatic session refresh
 */
export const authResponseInterceptor = async (
  response: Response,
  request: Request
): Promise<Response> => {
  // If response is OK, just pass it through
  if (response.ok) {
    return response;
  }

  // Only handle 401 errors
  if (response.status !== 401) {
    return response;
  }

  // Check if we have the session refresh capability
  const waitForSessionRefresh = (window as any).__waitForSessionRefresh;

  if (!waitForSessionRefresh || typeof waitForSessionRefresh !== 'function') {
    // Fallback to old behavior if session context is not available
    logout("Your session has expired. Please sign in again.");
    throw new Error('Authentication required');
  }

  // Prevent duplicate refresh attempts
  if (refreshInProgress) {
    await refreshPromise;
    // After refresh completes, verify session is restored
    if (!isAuthenticated()) {
      throw new Error('Session refresh failed - authentication not restored');
    }
    // Retry the original request
    return retry(request);
  }

  // Start refresh process
  refreshInProgress = true;
  refreshPromise = (async () => {
    try {
      console.log('API client: Session expired, attempting refresh for', request.url);
      await waitForSessionRefresh();

      // Give a moment for cookies to propagate
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Verify session is actually restored
      if (!isAuthenticated()) {
        throw new Error('Session refresh failed - authentication not restored');
      }

      console.log('API client: Session refreshed successfully');
    } catch (error) {
      console.error('API client: Session refresh failed:', error);
      throw error;
    } finally {
      refreshInProgress = false;
      refreshPromise = null;
    }
  })();

  await refreshPromise;

  // Retry the original request with the refreshed session
  return retry(request);
};

/**
 * Request interceptor to ensure credentials are always included
 */
export const credentialsInterceptor = async (
  request: Request
): Promise<Request> => {
  // Clone the request with credentials: 'include'
  const withCredentials = new Request(request, {
    credentials: 'include',
  });
  unsentCopies.set(withCredentials, withCredentials.clone());
  return withCredentials;
};
