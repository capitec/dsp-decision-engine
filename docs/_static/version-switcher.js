// Injects a version switcher dropdown into the Furo sidebar.
// Reads /versions.json (written by CI) — gracefully does nothing if absent.
(function () {
  const BASE = (function () {
    // Derive the site root from the current URL.
    // URLs are either  /index.html  (root/latest)
    //               or /0.1.0/index.html  (versioned)
    const parts = window.location.pathname.split('/').filter(Boolean);
    // If the first segment looks like a version, strip it.
    if (parts.length && /^\d+\.\d+/.test(parts[0])) {
      return window.location.origin + '/' + parts[0].split('/')[0].replace(parts[0], '');
    }
    return window.location.origin;
  })();

  // Determine which version we are currently viewing.
  const CURRENT = document.documentElement.dataset.content_root
    ? window.location.pathname.split('/').filter(Boolean)[0]
    : 'latest';

  fetch(BASE + '/versions.json')
    .then(r => r.ok ? r.json() : Promise.reject())
    .then(versions => {
      if (!versions.length) return;

      const select = document.createElement('select');
      select.title = 'Switch version';
      select.style.cssText = [
        'width:100%', 'margin:0.5rem 0', 'padding:0.3rem 0.5rem',
        'border-radius:4px', 'border:1px solid var(--color-foreground-border)',
        'background:var(--color-background-secondary)',
        'color:var(--color-foreground-primary)', 'font-size:0.85rem',
        'cursor:pointer',
      ].join(';');

      versions.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v === 'latest' ? BASE + '/' : BASE + '/' + v + '/';
        opt.textContent = v === 'latest' ? v : 'v' + v;
        // Mark active entry (match either exact version or 'latest' at root).
        if (v === CURRENT || (v === 'latest' && !/^\d/.test(CURRENT))) {
          opt.selected = true;
        }
        select.appendChild(opt);
      });

      select.addEventListener('change', () => {
        window.location.href = select.value;
      });

      // Furo sidebar: inject just before the first nav element.
      const sidebar = document.querySelector('.sidebar-drawer .sidebar-container');
      if (!sidebar) return;
      const label = document.createElement('div');
      label.style.cssText = 'padding:0 1rem; font-size:0.75rem; color:var(--color-foreground-muted); margin-top:0.75rem;';
      label.textContent = 'Version';
      sidebar.prepend(select);
      sidebar.prepend(label);
    })
    .catch(() => { /* versions.json absent — local build, no switcher */ });
})();
