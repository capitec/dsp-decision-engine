import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import { toast } from "@/hooks/use-toast";
import { Copy, Plus, Trash2, Eye, EyeOff, Shield, AlertTriangle } from "lucide-react";
import {
  listApiTokensApiV1AuthTokensGet,
  createApiTokenApiV1AuthTokensPost,
  updateApiTokenApiV1AuthTokensTokenIdPut,
  deleteApiTokenApiV1AuthTokensTokenIdDelete
} from "@/api";
import type { TokenResponse, TokenCreate, TokenScope } from "@/api/types.gen";

export const ApiTokensPage: React.FC = () => {
  const [tokens, setTokens] = useState<TokenResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Create token dialog state
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newTokenName, setNewTokenName] = useState('');
  const [newTokenScopes, setNewTokenScopes] = useState<TokenScope[]>(['read']);
  const [newTokenExpires, setNewTokenExpires] = useState('');

  // New token display state
  const [newToken, setNewToken] = useState<string | null>(null);
  const [newTokenVisible, setNewTokenVisible] = useState(false);

  // Delete confirmation dialog state
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [tokenToDelete, setTokenToDelete] = useState<{ id: string; name: string } | null>(null);

  useEffect(() => {
    fetchTokens();
  }, []);

  const fetchTokens = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await listApiTokensApiV1AuthTokensGet();
      if (response.data) {
        setTokens(response.data.tokens);
      }

    } catch (error: any) {
      console.error('Failed to fetch tokens:', error);
      setError('Failed to load API tokens');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateToken = async () => {
    if (!newTokenName.trim()) {
      toast({
        title: "Name required",
        description: "Please enter a name for your API token.",
        variant: "destructive",
      });
      return;
    }

    try {
      setCreating(true);

      const tokenData: TokenCreate = {
        name: newTokenName.trim(),
        scopes: newTokenScopes,
      };

      if (newTokenExpires) {
        tokenData.expires_at = new Date(newTokenExpires).toISOString();
      }

      const response = await createApiTokenApiV1AuthTokensPost({
        body: tokenData
      });

      if (response.data) {
        // Add new token to list
        setTokens(prev => [response.data!.token_info, ...prev]);

        // Show the new token
        setNewToken(response.data.token);

        // Reset form
        setNewTokenName('');
        setNewTokenScopes(['read']);
        setNewTokenExpires('');

        toast({
          title: "Token created",
          description: "Your API token has been created successfully.",
        });
      }

    } catch (error: any) {
      console.error('Failed to create token:', error);
      toast({
        title: "Creation failed",
        description: error.message || "Failed to create API token. Please try again.",
        variant: "destructive",
      });
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteTokenClick = (tokenId: string, tokenName: string) => {
    setTokenToDelete({ id: tokenId, name: tokenName });
    setDeleteDialogOpen(true);
  };

  const handleDeleteTokenConfirm = async () => {
    if (!tokenToDelete) return;

    try {
      await deleteApiTokenApiV1AuthTokensTokenIdDelete({
        path: { token_id: tokenToDelete.id }
      });

      setTokens(prev => prev.filter(t => t.id !== tokenToDelete.id));

      toast({
        title: "Token deleted",
        description: `Token "${tokenToDelete.name}" has been deleted.`,
      });

    } catch (error: any) {
      console.error('Failed to delete token:', error);
      toast({
        title: "Delete failed",
        description: error.message || "Failed to delete token. Please try again.",
        variant: "destructive",
      });
    } finally {
      setDeleteDialogOpen(false);
      setTokenToDelete(null);
    }
  };

  const handleToggleActive = async (token: TokenResponse) => {
    try {
      const response = await updateApiTokenApiV1AuthTokensTokenIdPut({
        path: { token_id: token.id },
        body: { is_active: !token.is_active }
      });

      if (response.data) {
        setTokens(prev => prev.map(t => t.id === token.id ? response.data! : t));

        toast({
          title: token.is_active ? "Token deactivated" : "Token activated",
          description: `Token "${token.name}" has been ${token.is_active ? 'deactivated' : 'activated'}.`,
        });
      }

    } catch (error: any) {
      console.error('Failed to toggle token:', error);
      toast({
        title: "Update failed",
        description: error.message || "Failed to update token. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleCopyToken = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast({
        title: "Copied to clipboard",
        description: "Token has been copied to your clipboard.",
      });
    } catch (error) {
      toast({
        title: "Copy failed",
        description: "Failed to copy token. Please copy it manually.",
        variant: "destructive",
      });
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleDateString();
  };

  const formatRelativeDate = (dateString: string | null) => {
    if (!dateString) return 'Never';
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    return formatDate(dateString);
  };

  const isTokenExpired = (expiresAt: string | null) => {
    if (!expiresAt) return false;
    return new Date(expiresAt) < new Date();
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div>Loading API tokens...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="w-96">
          <CardContent className="pt-6">
            <div className="text-center text-red-600">{error}</div>
            <Button onClick={fetchTokens} className="w-full mt-4">
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold">API Tokens</h1>
          <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="w-4 h-4 mr-2" />
                Create Token
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create API Token</DialogTitle>
                <DialogDescription>
                  Create a new API token to access your projects programmatically.
                </DialogDescription>
              </DialogHeader>
              
              <div className="space-y-4">
                <div>
                  <Label htmlFor="tokenName">Token Name</Label>
                  <Input
                    id="tokenName"
                    value={newTokenName}
                    onChange={(e) => setNewTokenName(e.target.value)}
                    placeholder="e.g., Jupyter Notebook, Production Script"
                  />
                  <p className="text-sm text-gray-500 mt-1">
                    Choose a descriptive name to help you identify this token later.
                  </p>
                </div>
                
                <div>
                  <Label>Permissions</Label>
                  <div className="mt-2 space-y-2">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="read"
                        checked={newTokenScopes.includes('read')}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setNewTokenScopes(prev => [...prev, 'read']);
                          } else {
                            setNewTokenScopes(prev => prev.filter(s => s !== 'read'));
                          }
                        }}
                      />
                      <Label htmlFor="read" className="text-sm">
                        Read - View projects and export configurations
                      </Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="write"
                        checked={newTokenScopes.includes('write')}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setNewTokenScopes(prev => [...prev, 'write']);
                          } else {
                            setNewTokenScopes(prev => prev.filter(s => s !== 'write'));
                          }
                        }}
                      />
                      <Label htmlFor="write" className="text-sm">
                        Write - Read permissions + create/modify projects
                      </Label>
                    </div>
                  </div>
                </div>
                
                <div>
                  <Label htmlFor="expires">Expiration (Optional)</Label>
                  <Input
                    id="expires"
                    type="date"
                    value={newTokenExpires}
                    onChange={(e) => setNewTokenExpires(e.target.value)}
                    min={new Date().toISOString().split('T')[0]}
                  />
                  <p className="text-sm text-gray-500 mt-1">
                    Leave empty for no expiration. We recommend setting an expiration date for security.
                  </p>
                </div>
                
                <div className="flex justify-end gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setCreateDialogOpen(false)}
                    disabled={creating}
                  >
                    Cancel
                  </Button>
                  <Button onClick={handleCreateToken} disabled={creating}>
                    {creating ? 'Creating...' : 'Create Token'}
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* New Token Display */}
        {newToken && (
          <Card className="mb-6 border-green-200 bg-green-50">
            <CardHeader>
              <CardTitle className="text-green-800 flex items-center">
                <Shield className="w-5 h-5 mr-2" />
                Token Created Successfully
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-white p-4 rounded border">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <Label className="text-sm font-medium">Your API Token</Label>
                    <div className="mt-1 font-mono text-sm bg-gray-100 p-2 rounded border">
                      {newTokenVisible ? newToken : '••••••••••••••••••••••••••••••••••••••••'}
                    </div>
                  </div>
                  <div className="flex gap-2 ml-4">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setNewTokenVisible(!newTokenVisible)}
                    >
                      {newTokenVisible ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleCopyToken(newToken)}
                    >
                      <Copy className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>
              <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded">
                <div className="flex items-start">
                  <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5 mr-2" />
                  <div className="text-sm text-amber-800">
                    <strong>Important:</strong> This token will only be shown once. Make sure to copy it now and store it securely.
                  </div>
                </div>
              </div>
              <div className="mt-4 flex justify-end">
                <Button variant="outline" onClick={() => setNewToken(null)}>
                  I've Saved My Token
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Tokens List */}
        <Card>
          <CardHeader>
            <CardTitle>Your API Tokens</CardTitle>
          </CardHeader>
          <CardContent>
            {tokens.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Shield className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p>No API tokens yet.</p>
                <p className="text-sm">Create your first token to get started.</p>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Token</TableHead>
                    <TableHead>Scopes</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Last Used</TableHead>
                    <TableHead>Expires</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {tokens.map((token) => (
                    <TableRow key={token.id}>
                      <TableCell className="font-medium">
                        {token.name}
                      </TableCell>
                      <TableCell>
                        <code className="text-sm bg-gray-100 px-2 py-1 rounded">
                          {token.token_prefix}...
                        </code>
                      </TableCell>
                      <TableCell>
                        <div className="flex gap-1">
                          {token.scopes.map((scope: TokenScope) => (
                            <Badge key={scope} variant="outline" className="text-xs">
                              {scope}
                            </Badge>
                          ))}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Badge 
                            className={
                              !token.is_active 
                                ? 'bg-gray-100 text-gray-800'
                                : isTokenExpired(token.expires_at)
                                  ? 'bg-red-100 text-red-800'
                                  : 'bg-green-100 text-green-800'
                            }
                          >
                            {!token.is_active 
                              ? 'Inactive'
                              : isTokenExpired(token.expires_at)
                                ? 'Expired'
                                : 'Active'
                            }
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell className="text-sm text-gray-600">
                        {formatRelativeDate(token.last_used_at)}
                      </TableCell>
                      <TableCell className="text-sm text-gray-600">
                        {token.expires_at ? formatDate(token.expires_at) : 'Never'}
                      </TableCell>
                      <TableCell>
                        <div className="flex gap-1">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleToggleActive(token)}
                            disabled={isTokenExpired(token.expires_at)}
                          >
                            {token.is_active ? 'Deactivate' : 'Activate'}
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleDeleteTokenClick(token.id, token.name)}
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        {/* Delete Confirmation Dialog */}
        <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete API Token</AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to delete the token "{tokenToDelete?.name}"? 
                This action cannot be undone and will immediately revoke access for any applications using this token.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel onClick={() => setTokenToDelete(null)}>
                Cancel
              </AlertDialogCancel>
              <AlertDialogAction 
                onClick={handleDeleteTokenConfirm}
                className="bg-red-600 hover:bg-red-700"
              >
                Delete Token
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
};