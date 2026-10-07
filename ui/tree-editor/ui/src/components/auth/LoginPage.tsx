import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Github, User, Key, Mail, Building2, Smartphone, Shield } from "lucide-react";

interface AuthMethod {
  type: string;
  enabled: boolean;
  display_name: string;
  icon_name: string;
  registration_enabled?: boolean;
  provider_key?: string; // For OIDC providers, identifies which provider
}

interface AuthConfig {
  methods: AuthMethod[];
}

// Component for provider icon with fallback
const ProviderIcon: React.FC<{ iconName: string }> = ({ iconName }) => {
  const [imageError, setImageError] = useState(false);
  const providerName = iconName.toLowerCase();
  
  // Always try to load a custom icon first, regardless of provider name
  if (!imageError) {
    return (
      <img 
        src={`/icons/providers/${providerName}.svg`}
        alt={`${iconName} logo`}
        className="h-5 max-w-8 object-contain"
        onError={() => setImageError(true)}
      />
    );
  }
  
  // Fall back to Lucide icon if custom icon fails to load
  return getLucideIcon(iconName);
};

// Helper function to get the OAuth provider icon
const getOAuthIcon = (iconName: string) => {
  return <ProviderIcon iconName={iconName} />;
};

// Fallback Lucide icons
const getLucideIcon = (iconName: string) => {
  switch (iconName.toLowerCase()) {
    case 'microsoft':
      return <Building2 className="w-4 h-4" />;
    case 'github':
      return <Github className="w-4 h-4" />;
    case 'google':
      return <Mail className="w-4 h-4" />;
    case 'apple':
      return <Smartphone className="w-4 h-4" />;
    case 'key':
    default:
      return <Key className="w-4 h-4" />;
  }
};

export const LoginPage: React.FC = () => {
  const [authConfig, setAuthConfig] = useState<AuthConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [serverError, setServerError] = useState<string | null>(null);
  const [showRegister, setShowRegister] = useState(false);
  const [credentials, setCredentials] = useState({
    email: '',
    password: '',
    name: ''
  });
  const [authError, setAuthError] = useState<string | null>(null);
  
  // Check if this is popup mode
  const isPopupMode = new URLSearchParams(window.location.search).get('redirect') === 'popup';

  useEffect(() => {
    fetchAuthConfig();
    
    // Check for logout message from sessionStorage
    const logoutMessage = sessionStorage.getItem('logout_message');
    if (logoutMessage) {
      setAuthError(logoutMessage);
      sessionStorage.removeItem('logout_message');
    }
  }, []);

  const fetchAuthConfig = async () => {
    try {
      console.log('Fetching auth config from /api/auth/config...');
      const response = await fetch('/api/auth/config', {
        redirect: 'manual', // Don't automatically follow redirects
        credentials: 'include'
      });
      console.log('Auth config response:', response.status, response.statusText, response.type);

      // If we got redirected (302, 303, 307, etc.), the response type will be 'opaqueredirect'
      // or status will be 0 in manual redirect mode
      if (response.type === 'opaqueredirect' || response.status === 0) {
        console.log('Auth config endpoint returned redirect - user not authenticated');
        // This is expected when not authenticated, just set empty config
        setAuthConfig({ methods: [] });
        setServerError(null);
        return;
      }

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      // Check if response is JSON before parsing
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        console.error('Auth config response is not JSON, content-type:', contentType);
        throw new Error('Server returned non-JSON response');
      }

      const config = await response.json();
      console.log('Auth config received:', config);
      console.log('OIDC methods found:', config.methods?.filter((m: AuthMethod) => m.type === 'oidc'));
      setAuthConfig(config);
      setServerError(null);
    } catch (error) {
      console.error('Failed to fetch auth config:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setServerError(`Unable to connect to server: ${errorMessage}`);
      // Set a default empty config to prevent errors
      setAuthConfig({ methods: [] });
    } finally {
      setLoading(false);
    }
  };

  const handleBasicLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    try {
      const formData = new URLSearchParams();
      formData.append('email', credentials.email);
      formData.append('password', credentials.password);

      const response = await fetch('/api/auth/basic/login', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData,
      });

      if (response.ok) {
        if (isPopupMode && window.opener) {
          // Send success message to parent window
          window.opener.postMessage({ type: 'auth-success' }, window.location.origin);
          window.close();
        } else {
          // Trigger a custom event to notify app of login
          window.dispatchEvent(new CustomEvent('auth-login'));
        }
      } else {
        const error = await response.json();
        setAuthError(error.detail || 'Login failed');
      }
    } catch (error) {
      console.error('Login error:', error);
      const errorMessage = 'Login failed - please try again';
      setAuthError(errorMessage);
      
      // Send error to parent window if in popup mode
      if (isPopupMode && window.opener) {
        window.opener.postMessage({ type: 'auth-error', message: errorMessage }, window.location.origin);
      }
    }
  };

  const handleBasicRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    try {
      const formData = new URLSearchParams();
      formData.append('email', credentials.email);
      formData.append('password', credentials.password);
      formData.append('name', credentials.name);

      const response = await fetch('/api/auth/basic/register', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData,
      });

      if (response.ok) {
        if (isPopupMode && window.opener) {
          // Send success message to parent window
          window.opener.postMessage({ type: 'auth-success' }, window.location.origin);
          window.close();
        } else {
          // Trigger a custom event to notify app of login
          window.dispatchEvent(new CustomEvent('auth-login'));
        }
      } else {
        const error = await response.json();
        setAuthError(error.detail || 'Registration failed');
      }
    } catch (error) {
      console.error('Registration error:', error);
      const errorMessage = 'Registration failed - please try again';
      setAuthError(errorMessage);
      
      // Send error to parent window if in popup mode
      if (isPopupMode && window.opener) {
        window.opener.postMessage({ type: 'auth-error', message: errorMessage }, window.location.origin);
      }
    }
  };

  const handleOidcLogin = (providerKey?: string) => {
    const redirectParam = isPopupMode ? '?redirect=popup' : '';
    
    if (providerKey) {
      window.location.href = `/api/auth/oidc/login/${providerKey}${redirectParam}`;
    } else {
      // Fallback for single provider configurations
      window.location.href = `/api/auth/oidc/login${redirectParam}`;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="w-96">
          <CardContent className="pt-6">
            <div className="text-center">Loading...</div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const basicAuthMethod = authConfig?.methods?.find(m => m.type === 'basic');
  const oidcMethods = authConfig?.methods?.filter(m => m.type === 'oidc') || [];
  
  // Check if this is a single OIDC provider scenario
  const isSingleOidcProvider = oidcMethods.length === 1 && !basicAuthMethod;

  return (
    <div className={`${isPopupMode ? 'min-h-screen' : 'min-h-screen'} flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100`}>
      <div className="w-full max-w-md mx-4">
        {/* Header section with branding - smaller in popup mode */}
        <div className="text-center mb-6">
          {!isPopupMode && (
            <img 
              src="/decider-icon.png" 
              alt="DSP Decider Logo" 
              className="w-24 h-24 mx-auto mb-4 object-contain"
            />
          )}
          <h1 className={`${isPopupMode ? 'text-xl' : 'text-2xl'} font-bold text-gray-900 mb-2`}>
            {isPopupMode ? 'Session Refresh' : 'DSP Decider'}
          </h1>
          <p className="text-gray-600">
            {isPopupMode 
              ? 'Please sign in to continue working'
              : isSingleOidcProvider 
                ? `Sign in with ${oidcMethods[0]?.display_name}`
                : showRegister ? 'Create your account' : 'Welcome back'
            }
          </p>
        </div>
        
        <Card className="shadow-xl border-0">
          <CardHeader className="pb-4">
            <CardTitle className="text-center text-lg font-medium">
              {showRegister ? 'Create Account' : 'Sign In'}
            </CardTitle>
          </CardHeader>
        <CardContent className="space-y-4">
          {serverError && (
            <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-md">
              <div className="flex">
                <div className="text-destructive">
                  <p className="text-sm font-medium">Server Unavailable</p>
                  <p className="text-sm text-destructive-foreground">{serverError}</p>
                  <button 
                    onClick={fetchAuthConfig}
                    className="mt-2 text-sm text-destructive hover:text-destructive/80 underline"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            </div>
          )}
          
          {authError && (
            <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-md">
              <div className="text-destructive">
                <p className="text-sm">{authError}</p>
              </div>
            </div>
          )}
          {oidcMethods.length > 0 && (
            <>
              <div className={`space-y-3 ${isSingleOidcProvider ? 'py-4' : ''}`}>
                {oidcMethods.map((oidcMethod, index) => (
                  <Button
                    key={oidcMethod.provider_key || index}
                    onClick={() => handleOidcLogin(oidcMethod.provider_key)}
                    variant={isSingleOidcProvider ? "default" : "outline"}
                    className={`w-full h-12 text-base font-medium ${
                      isSingleOidcProvider 
                        ? 'bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg' 
                        : ''
                    }`}
                  >
                    <span className="mr-3">{getOAuthIcon(oidcMethod.icon_name)}</span>
                    Continue with {oidcMethod.display_name}
                  </Button>
                ))}
              </div>
              {basicAuthMethod && <Separator className="my-6" />}
            </>
          )}

          {basicAuthMethod && (
            <>
              <form onSubmit={showRegister ? handleBasicRegister : handleBasicLogin} className="space-y-4">
                {showRegister && (
                  <div>
                    <Label htmlFor="name">Full Name</Label>
                    <Input
                      id="name"
                      type="text"
                      value={credentials.name}
                      onChange={(e) => setCredentials(prev => ({ ...prev, name: e.target.value }))}
                      required
                    />
                  </div>
                )}
                
                <div>
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    value={credentials.email}
                    onChange={(e) => setCredentials(prev => ({ ...prev, email: e.target.value }))}
                    required
                  />
                </div>
                
                <div>
                  <Label htmlFor="password">Password</Label>
                  <Input
                    id="password"
                    type="password"
                    value={credentials.password}
                    onChange={(e) => setCredentials(prev => ({ ...prev, password: e.target.value }))}
                    required
                  />
                </div>
                
                <Button type="submit" className="w-full">
                  {showRegister ? 'Create Account' : 'Sign In'}
                </Button>
              </form>

              {basicAuthMethod.registration_enabled && (
                <div className="text-center">
                  <Button
                    variant="link"
                    onClick={() => {
                      setShowRegister(!showRegister);
                      setAuthError(null);
                    }}
                  >
                    {showRegister 
                      ? 'Already have an account? Sign in' 
                      : 'Need an account? Register'
                    }
                  </Button>
                </div>
              )}
            </>
          )}

          {!basicAuthMethod && oidcMethods.length === 0 && (
            <div className="text-center text-gray-500">
              No authentication methods available
            </div>
          )}
          
        </CardContent>
      </Card>
      
      {/* Footer for branding */}
      {isSingleOidcProvider && (
        <div className="mt-8 text-center">
          <p className="text-sm text-gray-500">
            Decision Science Platform • Secure & Collaborative
          </p>
        </div>
      )}
      </div>
    </div>
  );
};