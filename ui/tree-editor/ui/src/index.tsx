import React from 'react';
import ReactDOM from 'react-dom/client';
import '@xyflow/react/dist/style.css';
import './xy-theme.css';
import './index.css';
import App from './components/core/App';
import reportWebVitals from './reportWebVitals';
import { loadLiveOps } from './lib/live-ops';

// Configure API client with auth interceptors and relative URLs
// This must be imported before any API calls are made
import './lib/api-client';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
// The live-sync ops are WebAssembly that stays synchronous for callers, so it is loaded before anything renders.
loadLiveOps().then(() =>
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  )
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
