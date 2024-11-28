import React from 'react';
import './App.css';
import Editor from '@components/editor/Editor';
import { Sidebar } from '@components/sidebar/Sidebar';
import { EditorProvider } from '@ctx/editor/EditorContext';

function App() {
  return (
    <div className="App">
      <EditorProvider>
        <Sidebar></Sidebar>
        <Editor></Editor>
      </EditorProvider>
    </div>
  );
}

export default App;
