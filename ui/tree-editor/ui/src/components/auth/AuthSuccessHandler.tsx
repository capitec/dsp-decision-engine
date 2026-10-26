import React, { useEffect } from 'react';
import { isAuthenticated } from '../../lib/auth';

/**
 * This component should be rendered on pages that handle authentication success
 * (like after OIDC redirect). It checks if we're in popup mode and sends the 
 * appropriate message to the parent window.
 */
export const AuthSuccessHandler: React.FC = () => {
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const isPopupMode = urlParams.get('redirect') === 'popup';
    
    // Check both URL parameter and if this window was opened by another window
    const isInPopup = isPopupMode || (window.opener && window.opener !== window);
    
    if (isInPopup && isAuthenticated()) {
      console.log('AuthSuccessHandler: Detected popup mode with successful authentication');
      
      try {
        // We're in popup mode and authentication was successful
        if (window.opener && !window.opener.closed) {
          window.opener.postMessage({ type: 'auth-success' }, window.location.origin);
          console.log('AuthSuccessHandler: Sent auth-success message to parent window');
        }
      } catch (error) {
        console.warn('AuthSuccessHandler: Failed to send message to parent:', error);
      }
      
      // Close the popup with a slight delay to ensure message is sent
      setTimeout(() => {
        try {
          window.close();
        } catch (error) {
          console.warn('AuthSuccessHandler: Failed to close popup window:', error);
        }
      }, 100);
    }
  }, []);

  return null; // This component doesn't render anything visible
};

/**
 * Higher-order component to wrap pages that might handle authentication success
 */
export const withAuthSuccessHandler = <P extends object>(
  Component: React.ComponentType<P>
): React.FC<P> => {
  return (props: P) => (
    <>
      <AuthSuccessHandler />
      <Component {...props} />
    </>
  );
};