export interface NodeData {
  name: string;
  documentation: string;
  dependencies: string[];
}

export interface EditorState {
    projectId: string | null;
    nodes: NodeData[];
    loading: boolean;
    error: string | null;
}
  
export type EditorAction = 
    | { type: 'SET_PROJECT_ID'; payload: string }
    | { type: 'FETCH_NODES_START' }
    | { type: 'FETCH_NODES_SUCCESS'; payload: any[] }
    | { type: 'FETCH_NODES_FAILURE'; payload: string }
    | { type: 'CLEAR_PROJECT' };