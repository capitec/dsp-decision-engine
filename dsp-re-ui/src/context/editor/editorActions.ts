import { EditorAction } from './editorTypes';
import axios from 'axios';


export const selectNode = (nodeId: string | null): EditorAction => ({
  type: 'SELECT_NODE',
  payload: nodeId
});

export const setProjectId = (projectId: string): EditorAction => ({
  type: 'SET_PROJECT_ID',
  payload: projectId
});

export const clearProject = (): EditorAction => ({
  type: 'CLEAR_PROJECT'
});

export const fetchProjectNodes = (projectId: string, dispatch: React.Dispatch<EditorAction>) => {
  dispatch({ type: 'FETCH_NODES_START' });

  axios.get(`/api/projects/${projectId}/nodes`).then((response => {
    dispatch({ 
      type: 'FETCH_NODES_SUCCESS', 
      payload: response.data 
    });
  })).catch(error => {
    dispatch({ 
      type: 'FETCH_NODES_FAILURE', 
      payload: error instanceof Error 
        ? error.message 
        : 'An unknown error occurred' 
    });
  })
};