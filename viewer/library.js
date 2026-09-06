// Browsing what a run recorded.
//
// A run directory holds one replay per recorded organism — the generation's top
// few plus a random sample — and the interesting ones are not always the fast
// ones. This module turns that directory into a sortable, groupable, filterable
// list so the outliers can be found and played without knowing a filename.
//
// It knows nothing about rendering. It hands a `File` back to whoever asked.
//
// # Why headers rather than whole files
//
// A replay is mostly trajectory: a few hundred kilobytes of poses behind a few
// kilobytes of description. Everything this list needs — identity, parentage,
// fitness, metrics, and the whole genome — sits before the `trace` key, so each
// file is read only up to there. Three hundred replays cost a few megabytes of
// reads instead of thirty, and the trajectory is loaded only for the one
// actually being watched.

/** Bytes read from each replay while looking for the end of its header. */
const HEADER_SLICE = 65536;

/** How many replay headers to read at once. */
const CONCURRENCY = 16;

/** Sort keys offered in the dropdown, and what the presets below reach for. */
const SORTS = [
  { key: 'fitness', label: 'fitness', of: (e) => e.fitness },
  { key: 'dx', label: 'distance along +X', of: (e) => e.m.displacement_x },
  { key: 'speed', label: 'mean speed', of: (e) => e.speed },
  { key: 'path', label: 'path length', of: (e) => e.m.path_length },
  { key: 'wander', label: 'wander (path − displacement)', of: (e) => e.wander },
  { key: 'upright', label: 'seconds upright', of: (e) => e.m.upright_seconds },
  { key: 'height', label: 'mean height', of: (e) => e.m.mean_height },
  { key: 'actuation', label: 'actuation (effort)', of: (e) => e.m.actuation },
  { key: 'economy', label: 'distance per effort', of: (e) => e.economy },
  { key: 'breaks', label: 'joints lost', of: (e) => e.breaks },
  { key: 'caution', label: 'caution', of: (e) => e.caution },
  { key: 'parts', label: 'part count', of: (e) => e.parts },
  { key: 'generation', label: 'generation', of: (e) => e.generation },
  { key: 'organism', label: 'organism id', of: (e) => e.id },
];

/** Ways to bucket the list. `of` returns the heading a row belongs under. */
const GROUPS = [
  { key: 'none', label: 'no grouping', of: null },
  { key: 'generation', label: 'by generation', of: (e) => `generation ${e.generation}` },
  {
    key: 'parent',
    label: 'by parent',
    of: (e) => (e.parents[0] ? `child of ${e.parents[0]}` : 'founder'),
  },
  { key: 'shapes', label: 'by shape mix', of: (e) => e.shapeLabel },
  { key: 'parts', label: 'by part count', of: (e) => `${e.parts} parts` },
  {
    key: 'broke',
    label: 'by whether a joint failed',
    of: (e) => (e.breaks > 0 ? 'lost a joint' : 'intact'),
  },
];

/**
 * One-click views onto the same list. These are shortcuts, not the only way in:
 * every sort key and grouping above stays available underneath, and the filter
 * box composes with all of them.
 */
const PRESETS = [
  { label: 'Best', sort: 'fitness', dir: -1 },
  { label: 'Worst', sort: 'fitness', dir: 1 },
  { label: 'Fastest', sort: 'speed', dir: -1 },
  { label: 'Most upright', sort: 'upright', dir: -1 },
  { label: 'Wanderers', sort: 'wander', dir: -1 },
  { label: 'Most efficient', sort: 'economy', dir: -1 },
  { label: 'Hardest working', sort: 'actuation', dir: -1 },
  { label: 'Broke a joint', sort: 'breaks', dir: -1, filter: 'broke' },
  { label: 'Most cautious', sort: 'caution', dir: -1 },
  { label: 'Latest', sort: 'generation', dir: -1 },
];

const el = (id) => document.getElementById(id);

/** Read `file` up to the start of its trajectory and parse that as JSON. */
async function readHeader(file) {
  for (const size of [HEADER_SLICE, file.size]) {
    const text = await file.slice(0, size).text();
    const cut = text.indexOf(',"trace":');
    if (cut > 0) return JSON.parse(`${text.slice(0, cut)}}`);
    if (size >= file.size) break;
  }
  // No `trace` key at all: not a replay we can draw.
  return null;
}

/** Compact description of what an organism is made of, e.g. "2 capsule, box". */
function describeShapes(genome) {
  const counts = new Map();
  for (const p of genome.parts || []) {
    const k = p.shape || 'box';
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  return (
    [...counts]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([k, n]) => (n > 1 ? `${n} ${k}` : k))
      .join(', ') || '—'
  );
}

function entryFrom(header, file) {
  const m = header.metrics || {};
  const duration = m.duration || 1;
  const dx = m.displacement_x || 0;
  const effort = m.actuation || 0;
  return {
    file,
    id: header.organism_id,
    generation: header.generation,
    parents: header.parents || [0, 0],
    fitness: header.fitness ?? 0,
    m,
    speed: (m.displacement || 0) / duration,
    // How far it strayed from a straight line: a tumbler racks this up, a walker
    // barely moves it.
    wander: (m.path_length || 0) - (m.displacement || 0),
    // Distance bought per unit of motor impulse. Guarded so an organism that
    // never actuated does not sort to the top on a division by zero.
    economy: effort > 1e-6 ? dx / effort : 0,
    parts: (header.genome?.parts || []).length,
    caution: header.genome?.caution ?? 0,
    shapeLabel: describeShapes(header.genome || {}),
    breaks: m.joints_lost || 0,
    diverged: !!m.diverged,
  };
}

export function initLibrary({ onSelect }) {
  let entries = [];
  let sortKey = 'fitness';
  let dir = -1;
  let groupKey = 'none';
  let filter = '';
  let activeId = null;
  let activePreset = 'Best';

  const list = el('lib-list');
  const sortSel = el('lib-sort');
  const groupSel = el('lib-group');
  const filterInput = el('lib-filter');
  const presets = el('lib-presets');

  for (const s of SORTS) sortSel.add(new Option(s.label, s.key));
  for (const g of GROUPS) groupSel.add(new Option(g.label, g.key));
  sortSel.value = sortKey;
  groupSel.value = groupKey;

  for (const p of PRESETS) {
    const b = document.createElement('button');
    b.className = 'chip';
    b.textContent = p.label;
    b.addEventListener('click', () => {
      activePreset = p.label;
      sortKey = p.sort;
      dir = p.dir;
      filter = p.filter || '';
      sortSel.value = sortKey;
      filterInput.value = filter;
      render();
    });
    presets.appendChild(b);
  }

  sortSel.addEventListener('change', () => {
    sortKey = sortSel.value;
    activePreset = null;
    render();
  });
  groupSel.addEventListener('change', () => {
    groupKey = groupSel.value;
    render();
  });
  filterInput.addEventListener('input', () => {
    filter = filterInput.value.trim().toLowerCase();
    activePreset = null;
    render();
  });

  function num(v, d = 2) {
    return typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—';
  }

  /**
   * Free-text match over everything a row displays, so "capsule", "690",
   * "broke" or a parent id all narrow the list without needing their own
   * control.
   */
  function matches(e) {
    if (!filter) return true;
    const hay = [
      e.id,
      `gen ${e.generation}`,
      e.shapeLabel,
      `${e.parts} parts`,
      e.parents.join(' '),
      e.breaks > 0 ? 'broke broken lost' : 'intact',
      e.diverged ? 'diverged' : '',
    ]
      .join(' ')
      .toLowerCase();
    return filter.split(/\s+/).every((term) => hay.includes(term));
  }

  function render() {
    for (const b of presets.children) b.classList.toggle('on', b.textContent === activePreset);

    const sort = SORTS.find((s) => s.key === sortKey) || SORTS[0];
    const group = GROUPS.find((g) => g.key === groupKey) || GROUPS[0];
    const rows = entries.filter(matches).sort((a, b) => {
      const d = (sort.of(a) ?? 0) - (sort.of(b) ?? 0);
      // Ties break on fitness so the order is total and stable between renders.
      return (d || (a.fitness - b.fitness)) * dir;
    });

    list.textContent = '';
    if (!entries.length) {
      list.innerHTML = '<div id="lib-empty">No run loaded.</div>';
      return;
    }
    if (!rows.length) {
      list.innerHTML = '<div id="lib-empty">Nothing matches that filter.</div>';
      return;
    }

    let heading = null;
    for (const e of rows) {
      if (group.of) {
        const h = group.of(e);
        if (h !== heading) {
          heading = h;
          const g = document.createElement('div');
          g.className = 'lib-group';
          g.textContent = h;
          list.appendChild(g);
        }
      }
      const item = document.createElement('div');
      item.className = 'lib-item' + (e.id === activeId ? ' on' : '');
      const broke = e.breaks > 0 ? ` · <span class="broke">lost ${e.breaks}</span>` : '';
      item.innerHTML =
        `<div class="lib-name">gen ${e.generation} · org ${e.id}</div>` +
        `<div class="lib-score">${num(e.fitness)}</div>` +
        `<div class="lib-meta">${num(e.m.displacement_x)} m · ${num(e.speed)} m/s · ` +
        `${num(e.m.upright_seconds, 1)} s up · ${e.parts} parts · ${e.shapeLabel}${broke}</div>`;
      item.addEventListener('click', () => {
        activeId = e.id;
        render();
        onSelect(e);
      });
      list.appendChild(item);
    }
  }

  return {
    /** Note which organism is on screen, so the list can highlight it. */
    setActive(id) {
      activeId = id;
      render();
    },
    /**
     * Ingest a directory chosen with a directory picker. Only the manifest and
     * the replays are read; the multi-gigabyte per-organism log is deliberately
     * left alone, because only recorded organisms can actually be played.
     */
    async open(fileList, onProgress) {
      const files = [...fileList];
      const manifestFile = files.find((f) => f.name === 'manifest.json');
      const replays = files
        .filter((f) => /replays[\\/][^\\/]+\.json$/.test(f.webkitRelativePath || f.name))
        .sort((a, b) => a.name.localeCompare(b.name));

      let manifest = null;
      if (manifestFile) {
        try {
          manifest = JSON.parse(await manifestFile.text());
        } catch {
          manifest = null;
        }
      }

      if (!replays.length) {
        el('lib-title').textContent = 'No replays found';
        el('lib-sub').textContent =
          'Choose a run directory — the one holding manifest.json and replays/.';
        entries = [];
        render();
        return 0;
      }

      el('lib-title').textContent =
        manifest?.experiment_name || manifest?.experiment_id || 'run';
      // Read headers a batch at a time. Strictly sequentially, a few hundred
      // replays take long enough to feel broken; the files are independent, so
      // there is no reason to wait on each one in turn.
      const found = new Array(replays.length).fill(null);
      let next = 0;
      let done = 0;
      const worker = async () => {
        for (;;) {
          const i = next++;
          if (i >= replays.length) return;
          try {
            const header = await readHeader(replays[i]);
            if (header) found[i] = entryFrom(header, replays[i]);
          } catch {
            // A truncated or half-written replay is skipped rather than
            // aborting the whole directory: a run killed mid-write is a normal
            // thing to want to look at.
          }
          onProgress?.(++done, replays.length);
        }
      };
      await Promise.all(
        Array.from({ length: Math.min(CONCURRENCY, replays.length) }, worker),
      );
      // Kept in filename order so the initial, unsorted list reads sensibly.
      entries = found.filter(Boolean);
      const gens = new Set(entries.map((e) => e.generation));
      el('lib-sub').textContent =
        `${entries.length} replays across ${gens.size} recorded generations` +
        (manifest?.experiment_id ? ` · ${manifest.experiment_id}` : '');
      render();
      return entries.length;
    },
  };
}
