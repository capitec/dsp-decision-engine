// Injects a version switcher dropdown at the bottom of the sidebar.
// Reads /versions.json (written by CI).
(function () {
  function siteRoot() {
    // sphinx-sitemap writes a <link rel="canonical"> with the full absolute URL of
    // the current page, e.g.:
    //   https://capitec.github.io/dsp-decision-engine/0.0.1/index.html
    // We strip the page filename and the versioned path segment to get the repo root.
    const canonical = document.querySelector('link[rel="canonical"]');
    if (canonical) {
      const url = new URL(canonical.href);
      const parts = url.pathname.replace(/\/$/, '').split('/').filter(Boolean);
      // Drop the filename (last segment) and optionally the version segment.
      parts.pop(); // filename
      if (parts.length && /^\d+\.\d+/.test(parts[parts.length - 1])) {
        parts.pop(); // versioned subdirectory
      }
      return url.origin + (parts.length ? '/' + parts.join('/') : '');
    }
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
