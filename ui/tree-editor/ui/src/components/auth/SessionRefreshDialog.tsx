import React, { useState, useCallback } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertTriangle, Loader2, RefreshCw, LogOut } from "lucide-react";
import { isAuthenticated } from "../../lib/auth";

interface SessionRefreshDialogProps {
  open: boolean;
  onClose: () => void;
  onSessionRestored: () => void;
  onForceLogout: () => void;
  message?: string;
}

export const SessionRefreshDialog: React.FC<SessionRefreshDialogProps> = ({
  open,
  onClose,
  onSessionRestored,
  onForceLogout,
  message = "Your session has expired. Please refresh your authentication to continue."
}) => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);

  const handleRefreshSession = useCallback(async () => {
    setIsRefreshing(true);
    setRefreshError(null);

    try {
      // Open login page in a popup window
      const popup = window.open(
        '/login?redirect=popup',
        'session-refresh',
        'width=500,height=600,scrollbars=yes,resizable=yes'
      );

      if (!popup) {
        throw new Error('Popup blocked. Please allow popups for this site.');
      }

      // Set up timeout for the entire operation (5 minutes)
      const operationTimeout = setTimeout(() => {
        if (!popup.closed) {
          popup.close();
        }
        window.removeEventListener('message', messageHandler);
        clearInterval(checkClosed);
        setRefreshError('Authentication timed out. Please try again.');
        setIsRefreshing(false);
      }, 300000); // 5 minutes

      // Listen for messages from the popup
      const messageHandler = (event: MessageEvent) => {
        if (event.origin !== window.location.origin) return;

        if (event.data.type === 'auth-success') {
          // Authentication was successful - clean up immediately
          clearTimeout(operationTimeout);
          clearInterval(checkClosed);
          popup.close();
          window.removeEventListener('message', messageHandler);
          
          // Give a moment for cookies to be set, then verify
          setTimeout(() => {
            if (isAuthenticated()) {
              onSessionRestored();
              onClose();
            } else {
              setRefreshError('Session refresh failed. Please try again.');
            }
            setIsRefreshing(false);
          }, 500); // Wait 500ms for cookies to propagate
        } else if (event.data.type === 'auth-error') {
          clearTimeout(operationTimeout);
          clearInterval(checkClosed);
          popup.close();
          window.removeEventListener('message', messageHandler);
          setRefreshError(event.data.message || 'Authentication failed');
          setIsRefreshing(false);
        }
      };

      window.addEventListener('message', messageHandler);

      // Check popup status more frequently and also check for successful auth
      const checkClosed = setInterval(() => {
        if (popup.closed) {
          clearTimeout(operationTimeout);
          clearInterval(checkClosed);
          window.removeEventListener('message', messageHandler);
          
          // Check if authentication succeeded even without message
          if (isAuthenticated()) {
            onSessionRestored();
            onClose();
          } else if (isRefreshing) {
            setRefreshError('Authentication was cancelled or failed');
          }
          setIsRefreshing(false);
        } else {
          // Also periodically check if auth succeeded (in case message was missed)
          if (isAuthenticated() && isRefreshing) {
            clearTimeout(operationTimeout);
            clearInterval(checkClosed);
            popup.close();
            window.removeEventListener('message', messageHandler);
            onSessionRestored();
            onClose();
            setIsRefreshing(false);
          }
        }
      }, 500); // Check every 500ms instead of 1000ms

    } catch (error) {
      setRefreshError(error instanceof Error ? error.message : 'Failed to open login popup');
      setIsRefreshing(false);
    }
  }, [onSessionRestored, onClose, isRefreshing]);

  const handleForceLogout = useCallback(() => {
    onForceLogout();
  }, [onForceLogout]);

  return (
    <Dialog open={open} onOpenChange={() => {}}>
      <DialogContent className="sm:max-w-md" onPointerDownOutside={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-amber-600">
            <AlertTriangle className="h-5 w-5" />
            Session Expired
          </DialogTitle>
          <DialogDescription>
            {message}
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4">
          {refreshError && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{refreshError}</AlertDescription>
            </Alert>
          )}
          
          <div className="flex flex-col gap-3">
            <Button
              onClick={handleRefreshSession}
              disabled={isRefreshing}
              className="w-full"
            >
              {isRefreshing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Refreshing Session...
                </>
              ) : (
                <>
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Refresh Session
                </>
              )}
            </Button>
            
            <Button
              variant="outline"
              onClick={handleForceLogout}
              disabled={isRefreshing}
              className="w-full"
            >
              <LogOut className="mr-2 h-4 w-4" />
              Sign Out & Lose Progress
            </Button>
          </div>
          
          <div className="text-xs text-gray-500 text-center">
            <p>Refreshing your session will preserve your current work.</p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};