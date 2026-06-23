// Injects a version switcher dropdown at the bottom of the sidebar.
// Reads /versions.json (written by CI).
(function () {
  function buildWrapper(versions, error) {
    const parts = window.location.pathname.split('/').filter(Boolean);
    const inVersionedPath = parts.length && /^\d+\.\d+/.test(parts[0]);
    const BASE = window.location.origin;
    const CURRENT = inVersionedPath ? parts[0] : 'latest';

    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'padding:0.75rem 1rem; border-top:1px solid var(--color-foreground-border); margin-top:0.5rem;';

    const label = document.createElement('div');
    label.textContent = 'Version';
    label.style.cssText = 'font-size:0.75rem; color:var(--color-foreground-muted); margin-bottom:0.25rem;';

    const select = document.createElement('select');
    select.title = 'Switch version';
    select.style.cssText = [
      'width:100%', 'padding:0.3rem 0.5rem',
      'border-radius:4px', 'border:1px solid var(--color-foreground-border)',
      'background:var(--color-background-secondary)',
      'color:var(--color-foreground-primary)', 'font-size:0.85rem',
      'cursor:pointer',
    ].join(';');

    if (error) {
      select.disabled = true;
      const opt = document.createElement('option');
      opt.textContent = 'Could not load versions';
      select.appendChild(opt);
    } else {
      versions.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v === 'latest' ? BASE + '/' : BASE + '/' + v + '/';
        opt.textContent = v === 'latest' ? 'latest' : 'v' + v;
        if (v === CURRENT) opt.selected = true;
        select.appendChild(opt);
      });
      select.addEventListener('change', () => { window.location.href = select.value; });
    }

    wrapper.appendChild(label);
    wrapper.appendChild(select);
    return wrapper;
  }

  function inject(versions, error) {
    const scroll =
      document.querySelector('.sidebar-scroll') ||
      document.querySelector('.sidebar-container') ||
      document.querySelector('[data-bd-component="sidebar-primary"]');
    if (!scroll) return;
    scroll.appendChild(buildWrapper(versions, error));
  }

  function run() {
    fetch(window.location.origin + '/versions.json')
      .then(r => r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)))
      .then(versions => inject(versions, false))
      .catch(() => inject([], true));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
  } else {
    run();
  }
})();
