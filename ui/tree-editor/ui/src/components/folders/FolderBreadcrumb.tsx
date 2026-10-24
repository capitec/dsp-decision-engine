import React from 'react';
import { ChevronRight, Home } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { Folder } from '@/types/folder-types';

interface FolderBreadcrumbProps {
  hierarchy: Folder[];
  onNavigate: (folderId: string | null) => void;
  loading?: boolean;
}

export const FolderBreadcrumb: React.FC<FolderBreadcrumbProps> = ({
  hierarchy,
  onNavigate,
  loading = false,
}) => {
  // For mobile: show only last 2 items with ellipsis
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;
  const displayHierarchy = isMobile && hierarchy.length > 2
    ? [hierarchy[hierarchy.length - 2], hierarchy[hierarchy.length - 1]]
    : hierarchy;

  const showEllipsis = isMobile && hierarchy.length > 2;

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-600 animate-pulse">
        <div className="h-4 w-16 bg-gray-200 rounded" />
        <ChevronRight className="h-4 w-4 text-gray-400" />
        <div className="h-4 w-24 bg-gray-200 rounded" />
      </div>
    );
  }

  return (
    <nav className="flex items-center gap-1 text-sm" aria-label="Breadcrumb">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => onNavigate(null)}
        className="h-8 px-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100"
        aria-label="Navigate to root"
      >
        <Home className="h-4 w-4" />
        <span className="ml-1 hidden sm:inline">All Projects</span>
      </Button>

      {showEllipsis && (
        <>
          <ChevronRight className="h-4 w-4 text-gray-400 flex-shrink-0" />
          <span className="text-gray-400 px-1">...</span>
        </>
      )}

      {displayHierarchy.map((folder, index) => {
        const isLast = index === displayHierarchy.length - 1;

        return (
          <React.Fragment key={folder.id}>
            <ChevronRight className="h-4 w-4 text-gray-400 flex-shrink-0" />

            {isLast ? (
              <span className="text-gray-900 font-medium px-2 truncate max-w-[200px]" aria-current="page">
                {folder.name}
              </span>
            ) : (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onNavigate(folder.id)}
                className="h-8 px-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 truncate max-w-[200px]"
                title={folder.name}
              >
                {folder.name}
              </Button>
            )}
          </React.Fragment>
        );
      })}
    </nav>
  );
};
