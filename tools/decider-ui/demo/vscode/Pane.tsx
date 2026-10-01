import { useState, type ReactNode } from "react";

// A stand-in for VS Code's collapsible sidebar section (what Explorer calls a "pane": the file
// tree, Outline, Timeline are each one). Real ones are user-reorderable and remember their
// collapsed state; this just has to look and click the same way.

export function Pane({ title, icon, defaultOpen = true, actions, children }: {
  title: string;
  icon?: string;
  defaultOpen?: boolean;
  actions?: ReactNode;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="pane">
      <div className="pane-header" role="button" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="pane-twisty">{open ? "▾" : "▸"}</span>
        {icon && <span className="pane-icon">{icon}</span>}
        <h3 className="pane-title">{title}</h3>
        {actions && (
          <div className="pane-actions" onClick={(e) => e.stopPropagation()}>
            {actions}
          </div>
        )}
      </div>
      {open && <div className="pane-body">{children}</div>}
    </div>
  );
}
