import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

const apiTarget = process.env.API_TARGET ?? 'http://localhost:8000'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      '@components': fileURLToPath(new URL('./src/components', import.meta.url)),
      '@ctx': fileURLToPath(new URL('./src/context', import.meta.url)),
    },
  },
  server: {
    port: 3000,
    allowedHosts: ['localhost', 'ui'], // `ui` is the compose service name
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: false,
        secure: false,
      },
      '/auth': {
        target: apiTarget,
        changeOrigin: false,
        secure: false,
        // configure: (proxy, options) => {
        //   proxy.on('proxyRes', (proxyRes, req, res) => {
        //     // Handle redirects by rewriting location header
        //     if (proxyRes.statusCode >= 300 && proxyRes.statusCode < 400) {
        //       const location = proxyRes.headers.location;
        //       if (location) {
        //         // If it's a relative redirect, make sure it goes to frontend port
        //         if (location.startsWith('/')) {
        //           proxyRes.headers.location = `http://localhost:3000${location}`;
        //         }
        //         // If it's absolute and points to 8000, change to 3000
        //         else if (location.includes('localhost:8000')) {
        //           proxyRes.headers.location = location.replace('localhost:8000', 'localhost:3000');
        //         }
        //       }
        //     }
        //   });
        // },
      },
    },
  },
  build: {
    outDir: 'build',
    sourcemap: true,
  },
})