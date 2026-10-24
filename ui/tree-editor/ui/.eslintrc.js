module.exports = {
  root: true,
  extends: [
    'react-app'
  ],
  env: {
    browser: true,
    es6: true,
    node: true,
    jest: true
  },
  settings: {
    react: {
      version: 'detect'
    }
  },
  rules: {
    // Disable some rules that might be causing issues
    'no-unused-vars': 'warn',
    '@typescript-eslint/no-unused-vars': 'warn'
  },
  overrides: [
    {
      files: ['**/*.test.{js,jsx,ts,tsx}'],
      env: {
        jest: true
      }
    }
  ]
};