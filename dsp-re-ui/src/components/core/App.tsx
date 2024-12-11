import React from 'react';
import './App.css';
import Editor from '@components/editor/Editor';
import { EditorProvider } from '@ctx/editor/EditorContext';
import { SidebarProvider, SidebarTrigger, SidebarInset } from "@/components/ui/sidebar"
import { AppSidebar } from '@components/sidebar/Sidebar';

function App() {
  return (
    <div className="App">
      
      <EditorProvider>
      <SidebarProvider>
        <SidebarInset>        
          <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">

          <SidebarTrigger className="-mr-1 ml-auto rotate-180" />
          </header>
          <Editor/>
        </SidebarInset>
        <AppSidebar />
      </SidebarProvider>
      </EditorProvider>
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