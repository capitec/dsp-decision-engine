/**
 * Background token refresh system that monitors store activity and proactively
 * refreshes authentication tokens before they expire.
 *
 * This system:
 * - Listens to zustand store changes to detect user activity
 * - Checks session expiry status on activity
 * - Silently refreshes tokens in the background when nearing expiry
 * - Uses debouncing and locking to prevent multiple simultaneous refreshes
 * - Fails gracefully without disrupting the user experience
 */

import { getSessionStatus, isAuthenticated } from './auth';

interface RefreshState {
  lastRefreshAttempt: number;
  refreshInProgress: boolean;
  refreshPromise: Promise<void> | null;
}

const state: RefreshState = {
  lastRefreshAttempt: 0,
  refreshInProgress: false,
  refreshPromise: null,
};

// Configuration
const DEBOUNCE_WINDOW_MS = 2000; // Don't attempt refresh if one was attempted in last 2s
const REFRESH_THRESHOLD_S = 900; // Start refreshing when within 15 minutes of expiry (matches backend default)
const MIN_REFRESH_INTERVAL_MS = 60000; // Minimum 1 minute between successful refreshes

/**
 * Check if session is close to expiring and needs refresh
 */
function needsRefresh(): boolean {
  if (!isAuthenticated()) {
    return false; // Already expired, let the auth interceptor handle it
  }

  const statusValue = getSessionStatus();
  if (!statusValue) {
    return false;
  }

  const expiryTimestamp = parseInt(statusValue, 10);
  if (isNaN(expiryTimestamp)) {
    return false;
  }

  const currentTimestamp = Math.floor(Date.now() / 1000);
  const timeUntilExpiry = expiryTimestamp - currentTimestamp;

  // Get threshold from backend settings (default to 900s / 15 min)
  const threshold = REFRESH_THRESHOLD_S;

  // Refresh if we're within the threshold
  return timeUntilExpiry > 0 && timeUntilExpiry <= threshold;
}

/**
 * Perform a silent background refresh by making a lightweight API call
 * This triggers the backend's rolling session refresh mechanism
 */
async function performBackgroundRefresh(): Promise<void> {
  // Check debounce window
  const now = Date.now();
  const timeSinceLastAttempt = now - state.lastRefreshAttempt;

  if (timeSinceLastAttempt < DEBOUNCE_WINDOW_MS) {
    console.log('[BackgroundRefresh] Skipping refresh - within debounce window');
    return;
  }

  // Check if refresh is already in progress
  if (state.refreshInProgress && state.refreshPromise) {
    console.log('[BackgroundRefresh] Refresh already in progress, waiting...');
    await state.refreshPromise;
    return;
  }

  // Start refresh
  state.lastRefreshAttempt = now;
  state.refreshInProgress = true;

  state.refreshPromise = (async () => {
    try {
      console.log('[BackgroundRefresh] Starting silent session refresh...');

      // Make a lightweight API call to trigger session renewal
      // Using /api/auth/config because it's public and lightweight
      const response = await fetch('/api/auth/config', {
        method: 'GET',
        credentials: 'include', // Essential: include cookies for session renewal
      });

      if (response.ok) {
        console.log('[BackgroundRefresh] Session refreshed successfully');
      } else {
        console.log('[BackgroundRefresh] Refresh request returned non-OK status:', response.status);
        // Don't throw - fail silently
      }
    } catch (error) {
      // Fail silently - we don't want to disrupt the user
      console.log('[BackgroundRefresh] Silent refresh failed:', error);
    } finally {
      state.refreshInProgress = false;
      state.refreshPromise = null;
    }
  })();

  await state.refreshPromise;
}

/**
 * Handle user activity detected from zustand store changes
 * Check if session needs refresh and trigger it if necessary
 */
async function handleUserActivity(): Promise<void> {
  // Quick checks first
  if (!needsRefresh()) {
    return;
  }

  // Don't spam refreshes
  const now = Date.now();
  const timeSinceLastAttempt = now - state.lastRefreshAttempt;

  if (timeSinceLastAttempt < MIN_REFRESH_INTERVAL_MS) {
    return;
  }

  // Trigger background refresh
  // Don't await - fire and forget for better UX
  performBackgroundRefresh().catch(() => {
    // Swallow errors - already logged in performBackgroundRefresh
  });
}

/**
 * Create a zustand middleware that monitors store activity and triggers
 * background token refresh when needed
 */
export function createBackgroundRefreshMiddleware<T extends object>(
  config: (set: any, get: any, api: any) => T
) {
  return (set: any, get: any, api: any): T => {
    const wrappedSet: typeof set = (...args: any[]) => {
      // Call original set
      const result = set(...args);

      // Trigger activity check (async, non-blocking)
      handleUserActivity().catch(() => {
        // Swallow errors
      });

      return result;
    };

    return config(wrappedSet, get, api);
  };
}

/**
 * Subscribe to a zustand store to monitor activity for background refresh
 * This is an alternative to the middleware approach for stores that already
 * have complex middleware chains
 */
export function subscribeToStoreActivity(store: any): () => void {
  return store.subscribe(() => {
    // On any state change, check if we need to refresh
    handleUserActivity().catch(() => {
      // Swallow errors
    });
  });
}
