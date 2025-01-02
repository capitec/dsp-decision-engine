import React from 'react';
import './App.css';
import Editor from '@components/editor/Editor';
import { SidebarProvider, SidebarTrigger, SidebarInset } from "@/components/ui/sidebar"
import { AppSidebar } from '@components/sidebar/Sidebar';
import { EditorProvider } from '@components/editor/EditorContext';
import {
  ReactFlowProvider,
} from '@xyflow/react';

function App() {
  return (
    <div className="App">
      
      <ReactFlowProvider>
      <EditorProvider>
      <SidebarProvider>
        <SidebarInset>        
          <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">

          <SidebarTrigger className="-mr-1 ml-auto rotate-180" />
          </header>
          <Editor projectId="00000000-0000-0000-0000-000000000000"/>
        </SidebarInset>
        <AppSidebar />
      </SidebarProvider>
      </EditorProvider>
      </ReactFlowProvider>
    </div>
  );
}

export default App;



{/* <AppSidebar />
<SidebarInset>
<main>
  <SidebarTrigger />
  <Editor/>
</main>
</SidebarInset> */}