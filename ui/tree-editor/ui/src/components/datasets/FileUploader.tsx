import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  Upload, 
  CloudUpload, 
  CheckCircle, 
  XCircle, 
  FileText,
  AlertCircle,
  X
} from 'lucide-react';
import { DataSourceInfo, UploadResponse } from '../../types/datasource-types';

interface FileUpload {
  id: string;
  file: File;
  status: 'pending' | 'uploading' | 'success' | 'error';
  progress: number;
  message?: string;
  tableName?: string;
}

interface FileUploaderProps {
  datasourceName: string;
  datasourceInfo: DataSourceInfo;
}

export const FileUploader: React.FC<FileUploaderProps> = ({
  datasourceName,
  datasourceInfo
}) => {
  const [uploads, setUploads] = useState<FileUpload[]>([]);
  const [dragActive, setDragActive] = useState(false);

  const updateUpload = (id: string, updates: Partial<FileUpload>) => {
    setUploads(prev => prev.map(upload => 
      upload.id === id ? { ...upload, ...updates } : upload
    ));
  };

  const removeUpload = (id: string) => {
    setUploads(prev => prev.filter(upload => upload.id !== id));
  };

  const uploadFile = async (fileUpload: FileUpload) => {
    const { id, file } = fileUpload;
    
    try {
      updateUpload(id, { status: 'uploading', progress: 0 });

      // Create form data
      const formData = new FormData();
      formData.append('file', file);

      // Upload file with progress tracking
      const response = await fetch(
        `/api/v1/datasources/${encodeURIComponent(datasourceName)}/upload`,
        {
          method: 'POST',
          body: formData,
          credentials: 'include',
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: UploadResponse = await response.json();

      if (result.success) {
        updateUpload(id, { 
          status: 'success', 
          progress: 100, 
          message: result.message,
          tableName: result.table_name 
        });
      } else {
        updateUpload(id, { 
          status: 'error', 
          progress: 0, 
          message: result.message 
        });
      }
    } catch (error: any) {
      updateUpload(id, { 
        status: 'error', 
        progress: 0, 
        message: error.message || 'Upload failed' 
      });
    }
  };

  const handleFiles = useCallback((files: FileList | null) => {
    if (!files) return;

    const newUploads: FileUpload[] = Array.from(files).map(file => ({
      id: `${file.name}-${Date.now()}-${Math.random()}`,
      file,
      status: 'pending',
      progress: 0,
    }));

    setUploads(prev => [...prev, ...newUploads]);

    // Start uploading each file
    newUploads.forEach(uploadFile);
  }, []);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  }, [handleFiles]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
  }, [handleFiles]);

  const getStatusIcon = (status: FileUpload['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-600" />;
      case 'uploading':
        return <Upload className="w-5 h-5 text-blue-600 animate-pulse" />;
      default:
        return <FileText className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: FileUpload['status']) => {
    switch (status) {
      case 'success':
        return 'text-green-600';
      case 'error':
        return 'text-red-600';
      case 'uploading':
        return 'text-blue-600';
      default:
        return 'text-gray-500';
    }
  };

  const formatFileSize = (bytes: number) => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  const pendingUploads = uploads.filter(u => u.status === 'pending' || u.status === 'uploading').length;
  const successfulUploads = uploads.filter(u => u.status === 'success').length;
  const failedUploads = uploads.filter(u => u.status === 'error').length;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center">
            <CloudUpload className="w-5 h-5 mr-2" />
            Upload to {datasourceInfo.display_name}
            {pendingUploads > 0 && (
              <Badge className="ml-2" variant="outline">
                {pendingUploads} uploading
              </Badge>
            )}
          </div>
          {uploads.length > 0 && (
            <Button 
              onClick={() => setUploads([])} 
              variant="outline" 
              size="sm"
            >
              Clear All
            </Button>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Upload Area */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive
              ? 'border-blue-400 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
        >
          <CloudUpload className={`mx-auto h-12 w-12 mb-4 ${
            dragActive ? 'text-blue-500' : 'text-gray-400'
          }`} />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {dragActive ? 'Drop files here' : 'Upload files'}
          </h3>
          <p className="text-gray-600 mb-4">
            Drag and drop files here, or click to select files
          </p>
          <input
            type="file"
            multiple
            onChange={handleFileSelect}
            className="hidden"
            id="file-upload"
            accept=".csv,.parquet,.json,.jsonl"
          />
          <label htmlFor="file-upload">
            <Button asChild>
              <span>
                <Upload className="w-4 h-4 mr-2" />
                Select Files
              </span>
            </Button>
          </label>
          <p className="text-sm text-gray-500 mt-2">
            Supported formats: CSV, Parquet, JSON, JSONL
          </p>
        </div>

        {/* Upload Summary */}
        {uploads.length > 0 && (
          <div className="grid grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{pendingUploads}</div>
              <div className="text-sm text-gray-600">Uploading</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{successfulUploads}</div>
              <div className="text-sm text-gray-600">Completed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{failedUploads}</div>
              <div className="text-sm text-gray-600">Failed</div>
            </div>
          </div>
        )}

        {/* Upload List */}
        {uploads.length > 0 && (
          <div className="space-y-3">
            <h4 className="font-medium text-gray-900">Upload Progress</h4>
            {uploads.map((upload) => (
              <div key={upload.id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(upload.status)}
                    <div>
                      <h5 className="font-medium text-gray-900">{upload.file.name}</h5>
                      <p className="text-sm text-gray-500">
                        {formatFileSize(upload.file.size)}
                        {upload.tableName && (
                          <span className="ml-2">
                            → Table: <code className="bg-gray-100 px-1 rounded">{upload.tableName}</code>
                          </span>
                        )}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`text-sm font-medium capitalize ${getStatusColor(upload.status)}`}>
                      {upload.status}
                    </span>
                    <Button 
                      onClick={() => removeUpload(upload.id)} 
                      variant="ghost" 
                      size="sm"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </div>

                {upload.status === 'uploading' && (
                  <Progress value={upload.progress} className="mb-2" />
                )}

                {upload.message && (
                  <Alert className={`mt-2 ${upload.status === 'error' ? 'border-red-200' : 'border-green-200'}`}>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription className="text-sm">
                      {upload.message}
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};