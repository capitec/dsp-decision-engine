export interface TreeOutput {
    [key: string]: string;
}

export interface SidebarGroupLabelProps {
children: React.ReactNode;
}

export interface SidebarGroupContentProps {
children: React.ReactNode;
className?: string;
}

export interface ProjectMetadata {
name: string;
description: string;
}

export interface TableEditorProps {
outputs: TreeOutput[];
onOutputsChange: (outputs: TreeOutput[]) => void;
}