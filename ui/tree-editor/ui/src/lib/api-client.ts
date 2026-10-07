/**
 * API client configuration
 *
 * Configures the generated @hey-api/openapi-ts client with:
 * - Relative URLs (empty baseUrl) so requests go to current origin
 * - Authentication interceptors for automatic token refresh on 401 responses
 *
 * This file must be imported early in the app (see index.tsx) to initialize
 * the client before any API calls are made.
 */

import { client } from '@/api/client.gen';
import { authResponseInterceptor, credentialsInterceptor } from './auth-interceptors';

// Configure client to use relative URLs (same as fetchJSON behavior)
// This ensures requests go to the current origin in both dev and production
client.setConfig({
  baseUrl: '', // Empty string = relative URLs
  throwOnError: true, // Throw errors instead of returning { error } object
});

// Set up authentication interceptors
client.interceptors.request.use(credentialsInterceptor);
client.interceptors.response.use(authResponseInterceptor);

/**
 * Authenticated API client with automatic token refresh
 *
 * The client is already configured - just import SDK functions from @/api
 * and they will automatically use this authenticated client.
 *
 * @example
 * ```ts
 * import { listProjectsApiV1ProjectsGet } from '@/api';
 *
 * const { data: projects } = await listProjectsApiV1ProjectsGet();
 * ```
 */
export const apiClient = client;

export default apiClient;

