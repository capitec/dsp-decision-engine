// Adds a version dropdown to the sidebar. CI writes /versions.json (strings,
// "latest" first), puts each release at /<version>/ and the newest build at the
// site root too. One build serves both places, so the location decides which.
(function () {
  const scope = window.__md_scope; // root of this build: /<site>/ or /<site>/<version>/
  const load = (url) => fetch(url).then((r) => (r.ok ? r.json() : Promise.reject()));

  function inject(versions, base, current) {
    const sidebar = document.querySelector('.md-sidebar--primary .md-sidebar__inner');
    if (!sidebar) return;

    const label = document.createElement('div');
    label.textContent = 'Version';
    label.style.cssText = 'font-size:.65rem;margin-bottom:.25rem;color:var(--md-default-fg-color--light)';

    const select = document.createElement('select');
    select.title = 'Switch version';
    select.style.cssText =
      'width:100%;padding:.3rem .5rem;font:inherit;font-size:.7rem;cursor:pointer;border-radius:4px;' +
      'color:var(--md-default-fg-color);background:var(--md-default-bg-color);' +
      'border:1px solid var(--md-default-fg-color--lighter)';
    versions.forEach((v) => {
      const href = new URL(v === 'latest' ? '.' : v + '/', base).href;
      const option = new Option(v === 'latest' ? 'latest' : 'v' + v, href);
      option.selected = v === current;
      select.add(option);
    });
    select.addEventListener('change', () => { window.location.href = select.value; });

    const box = document.createElement('div');
    box.style.cssText = 'padding:.75rem .8rem';
    box.append(label, select);
    sidebar.append(box);
  }

  // The root copy has versions.json beside it; a versioned copy has it one level up.
  load(new URL('versions.json', scope))
    .then((versions) => [versions, scope, 'latest'])
    .catch(() =>
      load(new URL('../versions.json', scope)).then((versions) => [
        versions,
        new URL('..', scope),
        scope.pathname.split('/').filter(Boolean).pop(),
      ])
    )
    .then(([versions, base, current]) => inject(versions, base, current))
    .catch(() => {}); // no versions.json (local preview): no dropdown
})();
