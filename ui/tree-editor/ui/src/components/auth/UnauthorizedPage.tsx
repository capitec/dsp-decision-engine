import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ShieldX, ArrowLeft } from "lucide-react";
import { useSearchParams } from "react-router-dom";

export const UnauthorizedPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const [detail, setDetail] = useState<string | null>(null);

  useEffect(() => {
    const errorDetail = searchParams.get('detail');
    if (errorDetail) {
      // Decode the URL-encoded detail parameter
      try {
        setDetail(decodeURIComponent(errorDetail));
      } catch {
        setDetail(errorDetail);
      }
    }
  }, [searchParams]);

  const handleBackToLogin = () => {
    window.location.href = '/api/auth/logout';
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-red-50 to-orange-100">
      <div className="w-full max-w-md mx-4">
        {/* Header section with branding */}
        <div className="text-center mb-8">
          <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-destructive/10 flex items-center justify-center">
            <ShieldX className="w-12 h-12 text-destructive" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Access Denied</h1>
          <p className="text-gray-600">
            You are not authorized to access this resource
          </p>
        </div>
        
        <Card className="shadow-xl border-0">
          <CardHeader className="pb-4">
            <CardTitle className="text-center text-lg font-medium text-destructive">
              Authentication Failed
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Always show error details section, but with fallback message */}
            <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-md">
              <div className="text-destructive">
                <p className="text-sm font-medium">Error Details:</p>
                <p className="text-sm text-destructive-foreground mt-1">
                  {detail || "Authentication failed - please try logging in again"}
                </p>
              </div>
            </div>
            
            <div className="text-center text-gray-500 text-xs">
              <p className="mb-2">This could be due to:</p>
              <ul className="space-y-1 text-left max-w-xs mx-auto">
                <li>• Invalid or expired session</li>
                <li>• Insufficient permissions</li>
                <li>• Authentication provider issues</li>
                <li>• Group membership requirements</li>
              </ul>
            </div>

            <Button 
              onClick={handleBackToLogin}
              className="w-full h-12 text-base font-medium"
              variant="default"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Login
            </Button>
          </CardContent>
        </Card>
        
        {/* Footer for branding */}
        <div className="mt-8 text-center">
          <p className="text-sm text-gray-500">
            Decision Science Platform • Secure & Collaborative
          </p>
        </div>
      </div>
    </div>
  );
};