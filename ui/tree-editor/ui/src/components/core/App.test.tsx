import React from 'react';
import { test } from 'vitest';
import { render } from '@testing-library/react';
import App from './App';

test('the app renders without throwing', () => {
  render(<App />);
});
