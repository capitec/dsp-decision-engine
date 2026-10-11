// Renders <div class="drawio" data-src="diagrams/x.drawio"></div> with the draw.io viewer.
// data-src is relative to the site root, so it works at any page depth and in every version.
(function () {
  function render() {
    const els = [...document.querySelectorAll('.drawio[data-src]')];
    Promise.all(els.map((el) =>
      fetch(new URL(el.dataset.src, window.__md_scope))
        .then((r) => (r.ok ? r.text() : Promise.reject()))
        .then((xml) => {
          el.dataset.mxgraph = JSON.stringify({
            xml, highlight: '#f97316', nav: true, resize: true,
            toolbar: 'zoom layers lightbox', edit: '_blank',
          });
          el.classList.add('mxgraph');
        })
        .catch(() => { el.textContent = 'Diagram could not be loaded.'; })
    )).then(() => window.GraphViewer && window.GraphViewer.processElements());
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', render);
  else render();
})();
