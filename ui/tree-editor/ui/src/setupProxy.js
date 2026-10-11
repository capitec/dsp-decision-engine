const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:8000',
      changeOrigin: false,
      // Keep the /api prefix when forwarding to backend
      pathRewrite: {
        '^/': '/api/'
      }
    })
  );
};