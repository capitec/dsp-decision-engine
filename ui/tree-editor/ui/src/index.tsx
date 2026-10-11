import React from 'react';
import ReactDOM from 'react-dom/client';
import '@xyflow/react/dist/style.css';
import './xy-theme.css';
import './index.css';
import App from './components/core/App';
import reportWebVitals from './reportWebVitals';
import { loadLiveOps } from './lib/live-ops';
import { LiveOpsFailedBanner } from './components/core/LiveOpsFailedBanner';

// Configure API client with auth interceptors and relative URLs
// This must be imported before any API calls are made
import './lib/api-client';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
// The live-sync ops are WebAssembly that stays synchronous for callers, so it is loaded before anything renders.
// Without it the editors still work, saving over REST.
loadLiveOps().then((error) => {
  if (error) console.error('live editing unavailable:', error);
  root.render(
    <React.StrictMode>
      <App />
      {error && <LiveOpsFailedBanner reason={error} />}
    </React.StrictMode>
  );
});

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
