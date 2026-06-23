// Injects a version switcher dropdown at the bottom of the sidebar.
// Reads /versions.json (written by CI).
(function () {
  function siteRoot() {
    // docs-config.js (written by conf.py build hook) sets this from html_baseurl.
    if (window.DOCS_BASE_URL) return window.DOCS_BASE_URL.replace(/\/$/, '');
    // Fallback for local dev: just use origin (no subpath).
    return window.location.origin;
  }

  function currentVersion(base) {
    const path = window.location.pathname;
    const after = path.startsWith(new URL(base + '/').pathname)
      ? path.slice(new URL(base + '/').pathname.length)
      : path.slice(1);
    const segment = after.split('/')[0];
    return segment && segment !== 'index.html' ? segment : 'latest';
  }

  function buildWrapper(base, current, versions, error) {
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
        opt.value = v === 'latest' ? base + '/' : base + '/' + v + '/';
        opt.textContent = v === 'latest' ? 'latest' : 'v' + v;
        if (v === current) opt.selected = true;
        select.appendChild(opt);
      });
      select.addEventListener('change', () => { window.location.href = select.value; });
    }

    wrapper.appendChild(label);
    wrapper.appendChild(select);
    return wrapper;
  }

  function inject(base, current, versions, error) {
    const scroll =
      document.querySelector('.sidebar-scroll') ||
      document.querySelector('.sidebar-container') ||
      document.querySelector('[data-bd-component="sidebar-primary"]');
    if (!scroll) return;
    scroll.appendChild(buildWrapper(base, current, versions, error));
  }

  function run() {
    const base = siteRoot();
    const current = currentVersion(base);
    fetch(base + '/versions.json')
      .then(r => r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)))
      .then(versions => inject(base, current, versions, false))
      .catch(() => inject(base, current, [], true));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
  } else {
    run();
  }
})();
