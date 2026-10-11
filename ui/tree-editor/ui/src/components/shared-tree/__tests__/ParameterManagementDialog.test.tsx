import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ParameterManagementDialog } from '../ParameterManagementDialog';
import { paramNameError } from '@/utils/param-names';

describe('paramNameError', () => {
  it.each(['rate', 'Rate_2', 'a'])('accepts %s', (name) => {
    expect(paramNameError(name)).toBeNull();
  });

  it.each(['', '  ', '1x', '_x', 'my rate', 'a-b', 'class', 'None', 'lambda'])('rejects "%s"', (name) => {
    expect(paramNameError(name)).not.toBeNull();
  });
});

describe('ParameterManagementDialog', () => {
  const setup = () => {
    const onSave = vi.fn();
    render(
      <ParameterManagementDialog
        open
        onOpenChange={() => {}}
        parameters={[{ id: 'p', name: 'rate', type: 'String', value: '' }]}
        onSave={onSave}
        isParameterInUse={() => false}
      />
    );
    return onSave;
  };

  it('refuses to save a renamed parameter that is not an identifier', () => {
    const onSave = setup();
    fireEvent.change(screen.getByDisplayValue('rate'), { target: { value: 'for' } });
    fireEvent.click(screen.getByRole('button', { name: /^save/i }));
    expect(onSave).not.toHaveBeenCalled();
    expect(screen.getAllByText(/Python keyword/).length).toBeGreaterThan(0);
  });

  it('trims names before saving', () => {
    const onSave = setup();
    fireEvent.change(screen.getByDisplayValue('rate'), { target: { value: '  new_rate  ' } });
    fireEvent.click(screen.getByRole('button', { name: /^save/i }));
    expect(onSave).toHaveBeenCalledWith([expect.objectContaining({ name: 'new_rate' })]);
  });
});
