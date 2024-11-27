import React from 'react';
import './App.css';
import Editor from '@components/editor/Editor';
import { EditorProvider } from '@ctx/editor/EditorContext';

function App() {
  return (
    <div className="App">
      <EditorProvider>
        <Editor projectId={"00000000-0000-0000-0000-000000000000"}></Editor>
      </EditorProvider>
    </div>
  );
}

export default App;
