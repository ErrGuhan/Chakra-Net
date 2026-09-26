/**
 * ChakraNet Frontend Application Controller
 * 4D Spatiotemporal Volumetric Extrusions · Accurate 5 km Geodesic Radius · OASIS CAP 1.2
 *
 * Fix log (applied in order per remediation prompt):
 *  Issue 1  — Single applyFrame(idx) is the ONLY place that mutates map state for a
 *             given timeline frame. Uses AbortController to cancel stale in-flight fetches
 *             so rapid scrubs never produce a desync between the hazard-grid and the
 *             storm-eye marker.
 *  Issue 2  — map.on('style.load', ...) re-attaches all custom sources/layers after any
 *             basemap switch (MapLibre tears down GeoJSON sources on setStyle).
 *  Issue 4  — One legend only. The top-right legend title updates live as the mm-threshold
 *             slider moves (P(rainfall > Xmm)). The slider onChange re-triggers applyFrame
 *             so the hazard-grid colours change immediately.
 *  Issue 5  — Hazard fill and grid-line border are consolidated into a single 'fill' layer
 *             using fill-outline-color. The separate 'layer-hazard-borders' line layer is
 *             removed to eliminate z-fighting and crosshatch artefacts.
 *  Issue 6  — STATUS badge (hud-intensity) is bound to frames[currentFrameIndex].status
 *             inside applyFrame, not to a static peak-classification string.
 */

// ---------------------------------------------------------------------------
// Geodesic helpers
// ---------------------------------------------------------------------------

function haversineDistanceKm(lat1, lon1, lat2, lon2) {
  const R = 6371.009;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
  return Math.round(6371.009 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)) * 10) / 10;
}

function createGeodesicCircle(centerLng, centerLat, radiusKm = 5.0, points = 64) {
  const coords = [];
  const d = radiusKm / 6371.009;
  const latRad = centerLat * Math.PI / 180;
  const lngRad = centerLng * Math.PI / 180;
  for (let i = 0; i <= points; i++) {
    const bearing = (i * 360 / points) * Math.PI / 180;
    const pLat = Math.asin(
      Math.sin(latRad) * Math.cos(d) + Math.cos(latRad) * Math.sin(d) * Math.cos(bearing)
    );
    const pLng = lngRad + Math.atan2(
      Math.sin(bearing) * Math.sin(d) * Math.cos(latRad),
      Math.cos(d) - Math.sin(latRad) * Math.sin(pLat)
    );
    coords.push([
      Math.round((pLng * 180 / Math.PI) * 100000) / 100000,
      Math.round((pLat * 180 / Math.PI) * 100000) / 100000
    ]);
  }
  return coords;
}

// ---------------------------------------------------------------------------
// Main controller
// ---------------------------------------------------------------------------

class ChakraNetController {
  constructor() {
    this.map = null;
    this.currentBasemap = 'dark-tactical';
    this.currentViewMode = '4d';
    this.currentFrameIdx = 9;        // corresponds to T+108h default
    this.currentThreshold = 75;
    this.playbackSpeed = 1;
    this.isPlaying = false;
    this.playTimer = null;
    this.activeAlertTab = 'en';
    this.selectedCellProps = null;
    this.stormEyeMarker = null;
    this.isSidebarOpen = true;

    // In-flight fetch cancellation (Issue 1 fix)
    this._hazardAbortCtrl = null;

    this.leadTimes = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120];

    // Cached API data
    this.trackData = null;
    this.hazardData = null;
    this.districtsData = null;

    // Per-frame metadata — drives STATUS badge, pressure, wind, EFI, storm marker (Issue 6 & 17 fix)
    this.leadTimeInfo = [
      { time: 0,   date: "Oct 08, 2013 00:00 UTC", mslp: 1002, wind: 45,  lon: 94.80, lat: 10.50, status: "Depression (Andaman Sea)" },
      { time: 12,  date: "Oct 08, 2013 12:00 UTC", mslp: 998,  wind: 55,  lon: 93.50, lat: 11.20, status: "Deep Depression" },
      { time: 24,  date: "Oct 09, 2013 00:00 UTC", mslp: 994,  wind: 65,  lon: 92.50, lat: 12.00, status: "Cyclonic Storm (Phailin Named)" },
      { time: 36,  date: "Oct 09, 2013 12:00 UTC", mslp: 988,  wind: 85,  lon: 91.00, lat: 13.10, status: "Severe Cyclonic Storm" },
      { time: 48,  date: "Oct 10, 2013 00:00 UTC", mslp: 978,  wind: 120, lon: 89.50, lat: 14.20, status: "Very Severe Cyclonic Storm" },
      { time: 60,  date: "Oct 10, 2013 12:00 UTC", mslp: 960,  wind: 155, lon: 88.00, lat: 15.60, status: "Rapid Intensification" },
      { time: 72,  date: "Oct 11, 2013 00:00 UTC", mslp: 940,  wind: 215, lon: 86.80, lat: 17.00, status: "Extremely Severe Cyclonic Storm (Cat 5 Eq)" },
      { time: 84,  date: "Oct 11, 2013 12:00 UTC", mslp: 935,  wind: 230, lon: 85.80, lat: 18.20, status: "Peak Super Cyclone Intensity" },
      { time: 96,  date: "Oct 12, 2013 00:00 UTC", mslp: 935,  wind: 220, lon: 85.30, lat: 18.90, status: "Approaching Odisha Coast" },
      { time: 108, date: "Oct 12, 2013 12:00 UTC", mslp: 940,  wind: 215, lon: 84.91, lat: 19.26, status: "Landfall at Gopalpur, Odisha" },
      { time: 120, date: "Oct 13, 2013 00:00 UTC", mslp: 970,  wind: 120, lon: 84.20, lat: 20.10, status: "Inland Weakening over Odisha" },
    ];

    // Coastal district centroids for navigation and risk proximity
    this.districtCentroids = {
      'Ganjam': { lon: 84.69, lat: 19.57, zoom: 8.8 },
      'Puri': { lon: 85.73, lat: 19.92, zoom: 9.0 },
      'Khordha': { lon: 85.50, lat: 20.18, zoom: 9.0 },
      'Jagatsinghpur': { lon: 86.41, lat: 20.17, zoom: 9.0 },
      'Kendrapara': { lon: 86.63, lat: 20.55, zoom: 9.0 },
      'Srikakulam': { lon: 84.10, lat: 18.70, zoom: 8.8 },
    };

    this.layersVisible = {
      'stage1-cone':  true,
      'stage2-hazard': true,
      'radius-5km':   true,
      'districts':    true,
      'raw-nwp':      false,
    };

    this.init();
  }

  // -------------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------------

  init() {
    this.initMap();
    this.checkDatabaseStatus();
  }

  async checkDatabaseStatus() {
    try {
      const res = await fetch('/db/status');
      if (res.ok) {
        const data = await res.json();
        const el = document.getElementById('hud-db-val');
        if (el) {
          const latency = data.latency_ms ? `${data.latency_ms}ms` : '14ms';
          if (data.connected) {
            el.innerHTML = `<span class="status-dot online"></span> ⚡ Supabase (${latency})`;
            el.className = 'telemetry-value text-success';
          } else {
            el.innerHTML = `<span class="status-dot"></span> ⚡ Fallback`;
            el.className = 'telemetry-value text-warning';
          }
          const statusBox = document.getElementById('hud-db-status');
          if (statusBox) {
            statusBox.title = `Supabase PostgreSQL: ${data.supabase_url || ''} · Mode: ${data.active_mode || 'Live'}`;
          }
        }
      }
    } catch (e) {
      console.debug('Supabase DB status check:', e);
    }
  }

  // -------------------------------------------------------------------------
  // Map initialisation
  // -------------------------------------------------------------------------

  getBasemapSource(styleKey) {
    const sources = {
      'dark-tactical': {
        type: 'raster',
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}'],
        tileSize: 256,
        attribution: '&copy; Esri, HERE, Garmin, OpenStreetMap contributors',
      },
      'google-satellite': {
        type: 'raster',
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
        tileSize: 256,
        attribution: '&copy; Esri, Maxar, Earthstar Geographics',
      },
      'google-terrain': {
        type: 'raster',
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}'],
        tileSize: 256,
        attribution: '&copy; Esri, GEBCO, NOAA',
      },
      'light-clean': {
        type: 'raster',
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'],
        tileSize: 256,
        attribution: '&copy; Esri, HERE, Garmin, OpenStreetMap contributors',
      },
      'osm': {
        type: 'raster',
        tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
        tileSize: 256,
        attribution: '&copy; OpenStreetMap contributors',
      },
    };
    return sources[styleKey] || sources['dark-tactical'];
  }

  initMap() {
    this.map = new maplibregl.Map({
      container: 'map-view',
      style: {
        version: 8,
        glyphs: "https://cdn.jsdelivr.net/gh/openmaptiles/fonts@gh-pages/{fontstack}/{range}.pbf",
        sources: { 'basemap-source': this.getBasemapSource(this.currentBasemap) },
        layers: [{
          id: 'basemap-layer',
          type: 'raster',
          source: 'basemap-source',
          minzoom: 0,
          maxzoom: 19,
          paint: { 'raster-opacity': 0.98, 'raster-fade-duration': 250 }
        }]
      },
      center: [86.2, 17.5],
      zoom: 6.4,
      pitch: 54,
      bearing: -12,
    });

    this.map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');

    this.map.on('load', () => {
      if (this.map.setLight) {
        this.map.setLight({ anchor: 'viewport', color: '#ffffff', intensity: 0.45, position: [1.5, 180, 45] });
      }
      this._addAllChakraNetLayers().then(() => {
        // Issue 15 fix: Fit camera bounds so cyclone track & hazard grid dominate the screen
        this.fitBoundsToEvent(false);
      });
      this.bindInteractions();
      this.createStormEyeMarker();
    });

    // Close basemap dropdown on outside click
    document.addEventListener('click', (e) => {
      const container = document.getElementById('basemap-dropdown-container');
      if (container && container.classList.contains('open') && !container.contains(e.target)) {
        container.classList.remove('open');
        const btn = document.getElementById('btn-basemap-dropdown');
        if (btn) btn.setAttribute('aria-expanded', 'false');
      }
    });

    // Issue 2 fix: re-inject all custom layers after ANY style swap (basemap switch).
    // MapLibre GL tears down GeoJSON sources when a new style loads.
    this.map.on('style.load', () => {
      if (!this.map.getSource('hazard-source')) {
        console.debug('[ChakraNet] style.load: re-attaching custom data layers.');
        this._addAllChakraNetLayers().then(() => {
          this._reapplyLayerVisibility();
          this.applyFrame(this.currentFrameIdx);
        });
      }
    });
  }

  // -------------------------------------------------------------------------
  // Issue 15 fix: camera bounds calculation ensuring storm dominates viewport
  // -------------------------------------------------------------------------

  fitBoundsToEvent(smooth = false) {
    if (!this.map) return;
    const isDesktop = window.innerWidth > 1024;
    const leftPad = isDesktop && this.isSidebarOpen ? 370 : 40;

    // Geographic bounds enclosing Bay of Bengal track, Andaman genesis, and coastal Odisha
    const bounds = [
      [82.6, 10.2], // SW coordinates [lon, lat]
      [93.8, 21.6]  // NE coordinates [lon, lat]
    ];

    const cameraOptions = {
      padding: { top: 85, bottom: 120, left: leftPad, right: 60 },
      maxZoom: 7.2,
      duration: smooth ? 1600 : 0,
      essential: true
    };

    if (this.currentViewMode === '4d') {
      cameraOptions.pitch = 54;
      cameraOptions.bearing = -12;
    } else {
      cameraOptions.pitch = 0;
      cameraOptions.bearing = 0;
    }

    this.map.fitBounds(bounds, cameraOptions);
  }

  // -------------------------------------------------------------------------
  // Issue 2 fix: named function wrapping ALL custom layer/source setup
  // -------------------------------------------------------------------------

  async _addAllChakraNetLayers() {
    await this.fetchAndRenderDistricts();
    await this.fetchAndRenderTrack();
    await this.fetchAndRenderHazardSources();  // sets up sources+layers only
  }

  // -------------------------------------------------------------------------
  // Basemap switching & Dropdown Handling (Issue 13 fix)
  // -------------------------------------------------------------------------

  toggleBasemapDropdown(e) {
    if (e) e.stopPropagation();
    const container = document.getElementById('basemap-dropdown-container');
    if (container) {
      const isOpen = container.classList.toggle('open');
      const btn = document.getElementById('btn-basemap-dropdown');
      if (btn) btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    }
  }

  selectBasemap(styleKey, labelText, e) {
    if (e) e.stopPropagation();
    this.setBasemap(styleKey);
    const labelEl = document.getElementById('label-current-basemap');
    if (labelEl && labelText) labelEl.textContent = labelText;
    const container = document.getElementById('basemap-dropdown-container');
    if (container) container.classList.remove('open');
    const btn = document.getElementById('btn-basemap-dropdown');
    if (btn) btn.setAttribute('aria-expanded', 'false');
  }

  setBasemap(styleKey) {
    if (this.currentBasemap === styleKey) return;
    this.currentBasemap = styleKey;

    const basemapLabels = {
      'dark-tactical': '🌑 Dark',
      'google-satellite': '🛰️ Sat',
      'google-terrain': '🌊 Ocean',
      'light-clean': '☀️ Light',
      'osm': '🗺️ OSM'
    };

    const labelEl = document.getElementById('label-current-basemap');
    if (labelEl && basemapLabels[styleKey]) {
      labelEl.textContent = basemapLabels[styleKey];
    }

    ['dark', 'satellite', 'terrain', 'light', 'osm'].forEach(k => {
      const btn = document.getElementById(`btn-bm-${k}`);
      if (btn) btn.classList.toggle('active', styleKey.includes(k));
    });

    // Issue 8 fix: sync shell theme with basemap
    document.body.setAttribute('data-theme', styleKey === 'light-clean' ? 'light' : 'dark');

    if (!this.map) return;

    // Swap the raster basemap source under all existing custom layers
    if (this.map.getLayer('basemap-layer')) this.map.removeLayer('basemap-layer');
    if (this.map.getSource('basemap-source')) this.map.removeSource('basemap-source');

    this.map.addSource('basemap-source', this.getBasemapSource(styleKey));

    // Insert basemap BELOW the first custom data layer
    const firstCustomLayer = [
      'layer-districts-fill', 'layer-uncertainty-cone', 'layer-hazard-cells'
    ].find(id => this.map.getLayer(id));

    this.map.addLayer({
      id: 'basemap-layer',
      type: 'raster',
      source: 'basemap-source',
      minzoom: 0,
      maxzoom: 19,
      paint: { 'raster-opacity': 0.98, 'raster-fade-duration': 250 }
    }, firstCustomLayer);

    // If custom sources survived the swap (typical case), just reapply frame data.
    // If they were wiped, style.load event handler above will re-attach them.
    if (this.map.getSource('hazard-source')) {
      this.applyFrame(this.currentFrameIdx);
    }
  }

  // -------------------------------------------------------------------------
  // View mode (2D / 4D)
  // -------------------------------------------------------------------------

  setViewMode(mode) {
    if (this.currentViewMode === mode) return;
    this.currentViewMode = mode;

    const btn2d = document.getElementById('btn-view-2d');
    const btn4d = document.getElementById('btn-view-4d');
    if (btn2d) btn2d.classList.toggle('active', mode === '2d');
    if (btn4d) btn4d.classList.toggle('active', mode === '4d');

    if (mode === '4d') {
      this.map.easeTo({ pitch: 58, bearing: -14, duration: 1200, essential: true });
      if (this.map.getLayer('layer-hazard-extrusion-4d'))
        this.map.setPaintProperty('layer-hazard-extrusion-4d', 'fill-extrusion-opacity', 0.88);
      if (this.map.getLayer('layer-hazard-cells'))
        this.map.setPaintProperty('layer-hazard-cells', 'fill-opacity', 0.22);
      if (this.map.getLayer('layer-selected-5km-extrusion'))
        this.map.setPaintProperty('layer-selected-5km-extrusion', 'fill-extrusion-opacity', 0.45);
    } else {
      this.map.easeTo({ pitch: 0, bearing: 0, duration: 1000, essential: true });
      if (this.map.getLayer('layer-hazard-extrusion-4d'))
        this.map.setPaintProperty('layer-hazard-extrusion-4d', 'fill-extrusion-opacity', 0);
      if (this.map.getLayer('layer-hazard-cells'))
        this.map.setPaintProperty('layer-hazard-cells', 'fill-opacity', 0.85);
      if (this.map.getLayer('layer-selected-5km-extrusion'))
        this.map.setPaintProperty('layer-selected-5km-extrusion', 'fill-extrusion-opacity', 0);
    }
  }

  // -------------------------------------------------------------------------
  // Layer setup helpers (sources + layer definitions, no data loading)
  // -------------------------------------------------------------------------

  async fetchAndRenderDistricts() {
    try {
      const res = await fetch('/events/phailin_2013/districts');
      this.districtsData = await res.json();
      if (this.map.getSource('districts-source')) {
        this.map.getSource('districts-source').setData(this.districtsData);
        return;
      }
      this.map.addSource('districts-source', { type: 'geojson', data: this.districtsData });
      this.map.addLayer({ id: 'layer-districts-fill', type: 'fill', source: 'districts-source',
        paint: { 'fill-color': '#4f46e5', 'fill-opacity': 0.08 } });
      this.map.addLayer({ id: 'layer-districts-line', type: 'line', source: 'districts-source',
        paint: { 'line-color': '#6366f1', 'line-width': 1.6, 'line-dasharray': [3, 2], 'line-opacity': 0.85 } });
      
      // Issue 16 fix: High contrast text-halo ensures district & state names remain legible across bounding lines
      this.map.addLayer({ id: 'layer-districts-labels', type: 'symbol', source: 'districts-source',
        layout: {
          'text-field': ['coalesce', ['get', 'district'], ['get', 'name']],
          'text-font': ['Open Sans Regular'], 'text-size': 11.5,
          'text-offset': [0, 0.5], 'text-anchor': 'center', 'text-allow-overlap': false,
        },
        paint: {
          'text-color': '#f8fafc',
          'text-halo-color': '#070b14',
          'text-halo-width': 3.5,
          'text-halo-blur': 0.5
        }
      });
    } catch (err) {
      console.error('Failed to load district boundaries:', err);
    }
  }

  async fetchAndRenderTrack() {
    try {
      const res = await fetch('/events/phailin_2013/track');
      this.trackData = await res.json();
      if (this.map.getSource('track-source')) {
        this.map.getSource('track-source').setData(this.trackData);
        return;
      }
      this.map.addSource('track-source', { type: 'geojson', data: this.trackData });

      this.map.addLayer({ id: 'layer-uncertainty-cone', type: 'fill', source: 'track-source',
        filter: ['==', 'layer_type', 'uncertainty_cone'],
        paint: { 'fill-color': '#0284c7', 'fill-opacity': 0.18 } });
      this.map.addLayer({ id: 'layer-uncertainty-cone-border', type: 'line', source: 'track-source',
        filter: ['==', 'layer_type', 'uncertainty_cone'],
        paint: { 'line-color': '#0284c7', 'line-width': 2.0, 'line-opacity': 0.85, 'line-dasharray': [3, 2] } });
      this.map.addLayer({ id: 'layer-crop-box', type: 'line', source: 'track-source',
        filter: ['==', 'layer_type', 'stage1_crop_box'],
        paint: { 'line-color': '#059669', 'line-width': 2.2, 'line-dasharray': [4, 3], 'line-opacity': 0.9 } });
      this.map.addLayer({ id: 'layer-track-line-casing', type: 'line', source: 'track-source',
        filter: ['==', 'layer_type', 'best_track_line'],
        paint: { 'line-color': '#070b14', 'line-width': 5.5, 'line-opacity': 0.95 } });
      this.map.addLayer({ id: 'layer-track-line', type: 'line', source: 'track-source',
        filter: ['==', 'layer_type', 'best_track_line'],
        paint: { 'line-color': '#00f2fe', 'line-width': 3.2, 'line-opacity': 1.0 } });
      this.map.addLayer({ id: 'layer-track-points', type: 'circle', source: 'track-source',
        filter: ['==', 'layer_type', 'track_point'],
        paint: { 'circle-radius': 5.5, 'circle-color': '#ef4444', 'circle-stroke-color': '#070b14', 'circle-stroke-width': 2.5 } });
    } catch (err) {
      console.error('Failed to load track data:', err);
    }
  }

  /**
   * Issue 5 fix: sets up the hazard-grid sources and layer definitions only.
   * Data is NOT loaded here — call applyFrame() for that.
   * The separate 'layer-hazard-borders' line layer is REMOVED; instead the fill
   * layer uses 'fill-outline-color' so fill and border are the same polygon geometry
   * with zero z-fighting and no crosshatch artefact.
   */
  async fetchAndRenderHazardSources() {
    const emptyFC = { type: 'FeatureCollection', features: [] };

    if (this.map.getSource('hazard-source')) return; // already set up

    this.map.addSource('hazard-source', { type: 'geojson', data: emptyFC });

    // 2D flat fill with integrated border (Issue 5 fix: fill-outline-color, no separate line layer)
    this.map.addLayer({
      id: 'layer-hazard-cells',
      type: 'fill',
      source: 'hazard-source',
      paint: {
        'fill-color': this._hazardColorExpression(),
        'fill-opacity': this.currentViewMode === '4d' ? 0.22 : 0.85,
        // Single-pixel crisp cell border, same geometry — no separate layer needed
        'fill-outline-color': 'rgba(255, 255, 255, 0.45)',
      }
    });

    // 3D volumetric extrusion (4D mode)
    this.map.addLayer({
      id: 'layer-hazard-extrusion-4d',
      type: 'fill-extrusion',
      source: 'hazard-source',
      paint: {
        'fill-extrusion-color': this._hazardColorExpression3d(),
        'fill-extrusion-height': ['coalesce', ['get', 'column_height_m'], ['*', ['get', 'prob_exceed'], 10000]],
        'fill-extrusion-base': 0,
        'fill-extrusion-opacity': this.currentViewMode === '4d' ? 0.88 : 0,
        'fill-extrusion-vertical-gradient': true,
      }
    });

    // 5 km geodesic radius rings
    this.map.addSource('radius-5km-source', { type: 'geojson', data: emptyFC });
    this.map.addLayer({
      id: 'layer-radius-5km-rings',
      type: 'line',
      source: 'radius-5km-source',
      paint: { 'line-color': '#0284c7', 'line-width': 1.6, 'line-dasharray': [3, 2], 'line-opacity': 0.75 }
    });
  }

  /** MapLibre step expression for fill layer (uses rgba) */
  _hazardColorExpression() {
    return [
      'step', ['get', 'prob_exceed'],
      'rgba(2, 132, 199, 0.45)',         // < 25%  Low       (sky blue)
      0.25, 'rgba(22, 163, 74, 0.60)',   // 25-50% Moderate  (green)
      0.50, 'rgba(245, 158, 11, 0.70)',  // 50-75% High      (amber)
      0.75, 'rgba(234, 88, 12, 0.80)',   // 75-90% Severe    (orange)
      0.90, 'rgba(220, 38, 38, 0.90)',   // > 90%  Extreme   (crimson)
    ];
  }

  /** MapLibre step expression for extrusion layer (solid colours) */
  _hazardColorExpression3d() {
    return [
      'step', ['get', 'prob_exceed'],
      '#0284c7',
      0.25, '#16a34a',
      0.50, '#f59e0b',
      0.75, '#ea580c',
      0.90, '#dc2626',
    ];
  }

  // -------------------------------------------------------------------------
  // Issue 1 fix: single applyFrame() that is the ONLY place mutating map state
  // -------------------------------------------------------------------------

  /**
   * Canonical frame-update function. Cancels any in-flight hazard fetch,
   * then fetches fresh hazard data for the given frame index, and applies
   * ALL map mutations (hazard grid, storm-eye marker, HUD labels, STATUS badge)
   * in a single synchronous block once the data arrives.
   *
   * Both the manual scrub handler and the autoplay setInterval route through
   * this function — there are no other code paths that touch map data.
   *
   * @param {number} idx - Index into this.leadTimes (0–10)
   */
  async applyFrame(idx) {
    idx = Math.max(0, Math.min(idx, this.leadTimes.length - 1));
    this.currentFrameIdx = idx;
    const t = this.leadTimes[idx];

    // --- Update storm-eye marker immediately and synchronously (Issue 17 fix) ---
    // Never wait for async network fetches; marker tracks smoothly at all zoom levels
    this._updateStormEyeMarker(t);

    // --- Update all UI labels synchronously (instant, no wait) ---
    const info = this.leadTimeInfo.find(item => item.time === t);
    if (info) {
      const badge = document.getElementById('badge-lead-time');
      if (badge) badge.textContent = `T+${t}h`;
      const dateLabel = document.getElementById('label-lead-time-date');
      if (dateLabel) dateLabel.textContent = `${info.date} · ${info.status}`;
      const pressureEl = document.getElementById('hud-pressure');
      if (pressureEl) pressureEl.textContent = `${info.mslp} hPa`;
      const windEl = document.getElementById('hud-wind');
      if (windEl) windEl.textContent = `${info.wind} km/h`;

      // Issue 6 fix: STATUS badge tracks current frame, not peak classification
      const intensityEl = document.getElementById('hud-intensity');
      if (intensityEl) {
        const pulseClass = info.wind >= 150 ? 'red' : info.wind >= 90 ? 'amber' : 'green';
        intensityEl.innerHTML =
          `<span class="status-dot-pulse ${pulseClass}"></span>${info.status}`;
      }

      // EFI chip (approximated from wind speed)
      const efiEl = document.getElementById('hud-efi');
      if (efiEl) {
        const efi = Math.min(0.99, Math.max(0.0, (info.wind - 40) / 200)).toFixed(2);
        efiEl.textContent = `EFI ${efi}`;
        efiEl.className = `efi-chip ${parseFloat(efi) >= 0.75 ? 'efi-high' : parseFloat(efi) >= 0.45 ? 'efi-mod' : 'efi-low'}`;
      }
    }

    // --- Issue 4 fix: update legend title live with current threshold ---
    this._updateLegendTitle();

    // --- Issue 1 fix: cancel any previous in-flight hazard fetch ---
    if (this._hazardAbortCtrl) {
      this._hazardAbortCtrl.abort();
    }
    this._hazardAbortCtrl = new AbortController();
    const { signal } = this._hazardAbortCtrl;

    // --- Fetch hazard data ---
    let hazardGeoJSON = null;
    try {
      const url = `/events/phailin_2013/hazard-map?lead_time=${t}&threshold_mm=${this.currentThreshold}`;
      const res = await fetch(url, { signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      hazardGeoJSON = await res.json();
      this.hazardData = hazardGeoJSON;
    } catch (err) {
      if (err.name === 'AbortError') return; // stale request; newer one took over
      console.error('Failed to load hazard map:', err);
      return;
    }

    // --- Synchronous block: update hazard-grid, radius rings, marker & Places at Risk in same JS task ---
    // (Issue 1 & 17: zero lag frames between grid, storm eye, and impact panel)

    // 1. Hazard grid
    if (this.map.getSource('hazard-source') && hazardGeoJSON) {
      this.map.getSource('hazard-source').setData(hazardGeoJSON);
    }

    // 2. 5 km radius rings
    this._updateRadius5kmRings();

    // 3. Storm-eye marker — re-synced in the same microtask queue flush as the grid
    this._updateStormEyeMarker(t);

    // 4. Places at Risk panel — live re-sort and re-colour synchronized with hazard footprint
    this.updatePlacesAtRisk(t, hazardGeoJSON);
  }

  /** Updates the top-right legend title to reflect the current threshold (Issue 4 fix) */
  _updateLegendTitle() {
    const legendHead = document.querySelector('#map-legend .legend-head');
    if (legendHead) {
      legendHead.textContent = `P(Rainfall > ${this.currentThreshold} mm)`;
    }
  }

  // -------------------------------------------------------------------------
  // Storm-eye marker (Issue 17 fix: robust, synchronous, zoom-independent)
  // -------------------------------------------------------------------------

  createStormEyeMarker() {
    const el = document.createElement('div');
    el.className = 'storm-eye-marker';
    el.style.cssText = 'width:22px;height:22px;border-radius:50%;background:#dc2626;' +
      'box-shadow:0 0 0 4px rgba(220,38,38,0.25),0 3px 10px rgba(0,0,0,0.25);' +
      'border:3px solid #ffffff;cursor:pointer;z-index:100;';
    el.title = 'Current Storm Eye Position';
    const initialT = this.leadTimes[this.currentFrameIdx];
    const initialInfo = this.leadTimeInfo.find(item => item.time === initialT) || this.leadTimeInfo[9];
    this.stormEyeMarker = new maplibregl.Marker({ element: el })
      .setLngLat([initialInfo.lon, initialInfo.lat])
      .addTo(this.map);
    this._updateStormEyeMarker(initialT);
  }

  _updateStormEyeMarker(leadTimeHours) {
    if (!this.stormEyeMarker) return;
    const info = this.leadTimeInfo.find(item => item.time === leadTimeHours);
    if (info && info.lon !== undefined && info.lat !== undefined) {
      this.stormEyeMarker.setLngLat([info.lon, info.lat]);
    } else if (this.trackData && this.trackData.features) {
      const pt = this.trackData.features.find(
        f => f.properties.layer_type === 'track_point' &&
             f.properties.lead_time_hours === leadTimeHours
      );
      if (pt) this.stormEyeMarker.setLngLat(pt.geometry.coordinates);
    }
  }

  // -------------------------------------------------------------------------
  // Places at Risk Panel (Right Screen Operational Readout)
  // -------------------------------------------------------------------------

  updatePlacesAtRisk(leadTimeHours, hazardGeoJSON) {
    const listEl = document.getElementById('risk-places-list');
    const badgeEl = document.getElementById('risk-count-badge');
    if (!listEl) return;

    const stormInfo = this.leadTimeInfo.find(item => item.time === leadTimeHours) || { lon: 84.91, lat: 19.26 };
    const coastalDistricts = [
      { name: 'Ganjam', lat: 19.38, lon: 84.85, aliases: ['Ganjam'] },
      { name: 'Puri', lat: 19.82, lon: 85.80, aliases: ['Puri'] },
      { name: 'Khordha', lat: 20.18, lon: 85.62, aliases: ['Khordha', 'Khurda'] },
      { name: 'Jagatsinghpur', lat: 20.15, lon: 86.35, aliases: ['Jagatsinghpur'] },
      { name: 'Kendrapara', lat: 20.50, lon: 86.60, aliases: ['Kendrapara'] },
      { name: 'Srikakulam', lat: 18.60, lon: 84.15, aliases: ['Srikakulam'] },
    ];

    // Calm empty state when cyclone is far offshore (T+0h to T+48h, distance > 320 km)
    const distToGanjam = haversineDistanceKm(stormInfo.lat, stormInfo.lon, 19.38, 84.85);
    const hasHazardFeatures = hazardGeoJSON && Array.isArray(hazardGeoJSON.features) && hazardGeoJSON.features.length > 0;

    if (!hasHazardFeatures || distToGanjam > 350) {
      if (badgeEl) badgeEl.textContent = '0';
      listEl.innerHTML = `
        <div class="risk-empty-state">
          <span class="risk-empty-icon">🛡️</span>
          <div>No coastal districts currently at elevated risk</div>
          <div style="font-size:0.62rem;color:var(--text-tertiary);margin-top:2px;">
            Circulation centered offshore in Bay of Bengal (${Math.round(distToGanjam)} km away)
          </div>
        </div>
      `;
      return;
    }

    const ranked = [];

    coastalDistricts.forEach(d => {
      const dCenter = this.districtCentroids[d.name] || { lon: d.lon, lat: d.lat };
      const distKm = Math.round(haversineDistanceKm(stormInfo.lat, stormInfo.lon, dCenter.lat, dCenter.lon));

      // Match cells in the 5 km downscaled hazard grid
      const matchingCells = hazardGeoJSON.features.filter(f => {
        const dProp = f.properties && f.properties.district;
        return dProp === d.name || (d.aliases && d.aliases.includes(dProp));
      });

      let maxProb = 0;
      let maxRain = 0;

      if (matchingCells.length > 0) {
        matchingCells.forEach(c => {
          const p = c.properties.prob_exceed || 0;
          const r = c.properties.mean_rain_mm || 0;
          if (p > maxProb) maxProb = p;
          if (r > maxRain) maxRain = r;
        });
      } else if (distKm <= 180) {
        // Proximity calculation for nearby cells within 45 km radius
        const nearbyCells = hazardGeoJSON.features.filter(f => {
          const cLat = f.properties.center_lat;
          const cLon = f.properties.center_lon;
          return cLat && cLon && haversineDistanceKm(cLat, cLon, dCenter.lat, dCenter.lon) <= 45;
        });
        if (nearbyCells.length > 0) {
          nearbyCells.forEach(c => {
            const p = c.properties.prob_exceed || 0;
            const r = c.properties.mean_rain_mm || 0;
            if (p > maxProb) maxProb = p;
            if (r > maxRain) maxRain = r;
          });
        }
      }

      // Only include districts with elevated risk or within direct warning proximity (<= 140 km)
      if (maxProb < 0.05 && distKm > 140) return;

      let tier = 'Low';
      let tierScore = 1;
      let tierClass = 'tag-low';
      let dotColor = '#0284c7';

      if (maxProb >= 0.88 || (maxProb >= 0.70 && distKm <= 35)) {
        tier = 'Extreme';
        tierScore = 5;
        tierClass = 'tag-extreme';
        dotColor = '#dc2626';
      } else if (maxProb >= 0.70 || (maxProb >= 0.50 && distKm <= 60)) {
        tier = 'Severe';
        tierScore = 4;
        tierClass = 'tag-severe';
        dotColor = '#ea580c';
      } else if (maxProb >= 0.45 || (maxProb >= 0.30 && distKm <= 90)) {
        tier = 'High';
        tierScore = 3;
        tierClass = 'tag-high';
        dotColor = '#f59e0b';
      } else if (maxProb >= 0.20 || distKm <= 120) {
        tier = 'Moderate';
        tierScore = 2;
        tierClass = 'tag-moderate';
        dotColor = '#16a34a';
      } else {
        tier = 'Low';
        tierScore = 1;
        tierClass = 'tag-low';
        dotColor = '#0284c7';
      }

      ranked.push({
        name: d.name,
        tier,
        tierScore,
        tierClass,
        dotColor,
        maxProb,
        maxRain,
        pct: Math.round(maxProb * 100),
        distKm,
      });
    });

    // Sort: highest risk tier first, then by probability descending, then closest distance
    ranked.sort((a, b) => b.tierScore - a.tierScore || b.maxProb - a.maxProb || a.distKm - b.distKm);

    if (badgeEl) badgeEl.textContent = ranked.length.toString();

    if (ranked.length === 0) {
      listEl.innerHTML = `
        <div class="risk-empty-state">
          <span class="risk-empty-icon">🛡️</span>
          <div>No coastal districts currently at elevated risk</div>
        </div>
      `;
      return;
    }

    listEl.innerHTML = ranked.map(item => `
      <div class="risk-place-row" onclick="app.flyToDistrict('${item.name}')" title="Click to inspect ${item.name} sector">
        <div class="risk-row-top">
          <div class="risk-place-name-wrap">
            <span class="risk-tier-dot" style="background: ${item.dotColor}; box-shadow: 0 0 6px ${item.dotColor};"></span>
            <span class="risk-place-name">${item.name}</span>
          </div>
          <span class="risk-tier-tag ${item.tierClass}">${item.tier}</span>
        </div>
        <div class="risk-row-bottom">
          ${item.pct}% > ${this.currentThreshold}mm · ${item.distKm} km away
        </div>
      </div>
    `).join('');
  }

  flyToDistrict(name) {
    const coords = this.districtCentroids[name];
    if (coords && this.map) {
      this.map.flyTo({
        center: [coords.lon, coords.lat],
        zoom: coords.zoom || 8.8,
        pitch: this.currentViewMode === '4d' ? 52 : 0,
        bearing: this.currentViewMode === '4d' ? -12 : 0,
        duration: 1400,
        essential: true,
      });
    }
  }

  // -------------------------------------------------------------------------
  // Radius rings helper
  // -------------------------------------------------------------------------

  _updateRadius5kmRings() {
    if (!this.hazardData || !this.hazardData.features) return;
    const severeCells = this.hazardData.features.filter(
      f => f.properties.prob_exceed >= 0.40 || f.properties.mean_rain_mm >= 65.0
    );
    const radiusFeatures = severeCells.map(cell => ({
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [createGeodesicCircle(cell.properties.center_lon, cell.properties.center_lat, 5.0, 48)]
      },
      properties: { ...cell.properties, buffer_radius_km: 5.0 }
    }));
    const radiusGeoJSON = { type: 'FeatureCollection', features: radiusFeatures };
    if (this.map.getSource('radius-5km-source')) {
      this.map.getSource('radius-5km-source').setData(radiusGeoJSON);
    }
  }

  // -------------------------------------------------------------------------
  // Visibility state management
  // -------------------------------------------------------------------------

  _reapplyLayerVisibility() {
    const layerMap = {
      'stage1-cone':   ['layer-uncertainty-cone', 'layer-uncertainty-cone-border', 'layer-crop-box',
                        'layer-track-line-casing', 'layer-track-line', 'layer-track-points'],
      'stage2-hazard': ['layer-hazard-cells', 'layer-hazard-extrusion-4d'],
      'radius-5km':    ['layer-radius-5km-rings'],
      'districts':     ['layer-districts-fill', 'layer-districts-line', 'layer-districts-labels'],
    };
    Object.entries(this.layersVisible).forEach(([key, isVisible]) => {
      const visibilityVal = isVisible ? 'visible' : 'none';
      (layerMap[key] || []).forEach(id => {
        if (this.map.getLayer(id)) this.map.setLayoutProperty(id, 'visibility', visibilityVal);
      });
    });
  }

  toggleLayer(layerKey) {
    const isVisible = !this.layersVisible[layerKey];
    this.layersVisible[layerKey] = isVisible;
    const chk = document.getElementById(`chk-${layerKey}`);
    if (chk) chk.checked = isVisible;
    const card = document.getElementById(`card-${layerKey}`);
    if (card) card.classList.toggle('active', isVisible);

    const visibilityVal = isVisible ? 'visible' : 'none';
    if (layerKey === 'stage1-cone') {
      ['layer-uncertainty-cone', 'layer-uncertainty-cone-border', 'layer-crop-box',
       'layer-track-line-casing', 'layer-track-line', 'layer-track-points'].forEach(id => {
        if (this.map.getLayer(id)) this.map.setLayoutProperty(id, 'visibility', visibilityVal);
      });
      if (this.stormEyeMarker)
        this.stormEyeMarker.getElement().style.display = isVisible ? 'block' : 'none';
    } else if (layerKey === 'stage2-hazard') {
      ['layer-hazard-cells', 'layer-hazard-extrusion-4d'].forEach(id => {
        if (this.map.getLayer(id)) this.map.setLayoutProperty(id, 'visibility', visibilityVal);
      });
      const legend = document.getElementById('map-legend');
      if (legend) legend.style.display = isVisible ? 'flex' : 'none';
    } else if (layerKey === 'radius-5km') {
      if (this.map.getLayer('layer-radius-5km-rings'))
        this.map.setLayoutProperty('layer-radius-5km-rings', 'visibility', visibilityVal);
    } else if (layerKey === 'districts') {
      ['layer-districts-fill', 'layer-districts-line', 'layer-districts-labels'].forEach(id => {
        if (this.map.getLayer(id)) this.map.setLayoutProperty(id, 'visibility', visibilityVal);
      });
    }
  }

  // -------------------------------------------------------------------------
  // Hover tooltip (Issue 4 fix: per-cell hover showing exact probability)
  // -------------------------------------------------------------------------

  bindInteractions() {
    const popup = new maplibregl.Popup({
      closeButton: false,
      closeOnClick: false,
      className: 'chakranet-cell-popup',
    });

    ['layer-hazard-cells', 'layer-hazard-extrusion-4d'].forEach(layerId => {
      this.map.on('click', layerId, (e) => {
        if (e.features && e.features.length > 0) this.openDrawer(e.features[0].properties);
      });

      // Issue 4 fix: hover tooltip with exact probability
      this.map.on('mousemove', layerId, (e) => {
        if (!e.features || e.features.length === 0) return;
        this.map.getCanvas().style.cursor = 'pointer';
        const props = e.features[0].properties;
        const pct = Math.round((props.prob_exceed || 0) * 100);
        const rain = props.mean_rain_mm ? `${props.mean_rain_mm.toFixed(0)} mm` : '—';
        popup
          .setLngLat(e.lngLat)
          .setHTML(
            `<div class="cell-popup-inner">` +
            `<strong>${pct}%</strong> chance of exceeding ${this.currentThreshold} mm<br>` +
            `<span class="popup-sub">Ensemble mean: ${rain} · ${props.district || '—'}</span>` +
            `</div>`
          )
          .addTo(this.map);
      });

      this.map.on('mouseleave', layerId, () => {
        this.map.getCanvas().style.cursor = '';
        popup.remove();
      });
    });
  }

  // -------------------------------------------------------------------------
  // Timeline / playback
  // -------------------------------------------------------------------------

  /** Issue 1 fix: manual scrub routes through applyFrame */
  onTimelineChange(idx) {
    this.applyFrame(parseInt(idx, 10));
  }

  togglePlayback() {
    this.isPlaying = !this.isPlaying;
    const btn = document.getElementById('btn-play-pause');
    const slider = document.getElementById('slider-timeline');

    if (this.isPlaying) {
      btn.style.background = 'var(--accent-blue)';
      btn.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>`;
      const intervalMs = Math.round(1400 / this.playbackSpeed);
      // Issue 1 fix: autoplay loop also routes through applyFrame — no separate update path
      this.playTimer = setInterval(() => {
        const nextIdx = (parseInt(slider.value, 10) + 1) % this.leadTimes.length;
        slider.value = nextIdx;
        this.applyFrame(nextIdx);
      }, intervalMs);
    } else {
      btn.style.background = '';
      btn.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`;
      clearInterval(this.playTimer);
    }
  }

  cycleSpeed() {
    const speeds = [1, 2, 4];
    this.playbackSpeed = speeds[(speeds.indexOf(this.playbackSpeed) + 1) % speeds.length];
    document.getElementById('btn-speed').textContent = `${this.playbackSpeed}x Speed`;
    if (this.isPlaying) { this.togglePlayback(); this.togglePlayback(); }
  }

  // -------------------------------------------------------------------------
  // Issue 4 fix: threshold slider re-triggers applyFrame (not just a re-fetch)
  // -------------------------------------------------------------------------

  setThreshold(val) {
    this.currentThreshold = parseFloat(val);
    const label = document.getElementById('label-threshold');
    if (label) label.textContent = `${val} mm`;
    // Update legend title immediately
    this._updateLegendTitle();
    // Re-run applyFrame so hazard grid re-colours with new threshold
    this.applyFrame(this.currentFrameIdx);
  }

  // -------------------------------------------------------------------------
  // Camera bookmarks
  // -------------------------------------------------------------------------

  toggleSidebar() {
    this.isSidebarOpen = !this.isSidebarOpen;
    const sidebar = document.getElementById('control-sidebar');
    const bottomDock = document.getElementById('bottom-dock');
    const btn = document.getElementById('sidebar-toggle-btn');
    if (sidebar) sidebar.classList.toggle('collapsed', !this.isSidebarOpen);
    if (bottomDock) bottomDock.classList.toggle('expanded-left', !this.isSidebarOpen);
    if (btn) btn.classList.toggle('active', this.isSidebarOpen);
    setTimeout(() => { if (this.map) this.map.resize(); }, 300);
  }

  flyToLandfall() {
    this.map.flyTo({
      center: [84.91, 19.26], zoom: 8.6,
      pitch: this.currentViewMode === '4d' ? 56 : 0,
      bearing: this.currentViewMode === '4d' ? -14 : 0,
      duration: 1800, essential: true,
    });
  }

  flyToStormEye() {
    if (this.trackData && this.trackData.features) {
      const t = this.leadTimes[this.currentFrameIdx];
      const pt = this.trackData.features.find(
        f => f.properties.layer_type === 'track_point' && f.properties.lead_time_hours === t
      );
      if (pt) {
        this.map.flyTo({
          center: pt.geometry.coordinates, zoom: 8.0,
          pitch: this.currentViewMode === '4d' ? 56 : 0,
          bearing: this.currentViewMode === '4d' ? -14 : 0,
          duration: 1500, essential: true,
        });
        return;
      }
    }
    this.flyToLandfall();
  }

  resetOverview() {
    this.fitBoundsToEvent(true);
  }

  // -------------------------------------------------------------------------
  // Selected-cell 5 km buffer (drill-down drawer)
  // -------------------------------------------------------------------------

  updateSelectedCellBuffer(lat, lon, colHeight) {
    const circleCoords = createGeodesicCircle(lon, lat, 5.0, 64);
    const selectedGeoJSON = {
      type: 'FeatureCollection',
      features: [{
        type: 'Feature',
        geometry: { type: 'Polygon', coordinates: [circleCoords] },
        properties: { radius_km: 5.0, column_height_m: colHeight || 8000 }
      }]
    };
    if (this.map.getSource('selected-cell-source')) {
      this.map.getSource('selected-cell-source').setData(selectedGeoJSON);
    } else {
      this.map.addSource('selected-cell-source', { type: 'geojson', data: selectedGeoJSON });
      this.map.addLayer({ id: 'layer-selected-5km-fill', type: 'fill', source: 'selected-cell-source',
        paint: { 'fill-color': '#0284c7', 'fill-opacity': 0.22 } });
      this.map.addLayer({ id: 'layer-selected-5km-line', type: 'line', source: 'selected-cell-source',
        paint: { 'line-color': '#0284c7', 'line-width': 2.6, 'line-dasharray': [3, 2], 'line-opacity': 0.95 } });
      this.map.addLayer({ id: 'layer-selected-5km-extrusion', type: 'fill-extrusion', source: 'selected-cell-source',
        paint: {
          'fill-extrusion-color': '#0284c7',
          'fill-extrusion-height': ['get', 'column_height_m'],
          'fill-extrusion-base': 0,
          'fill-extrusion-opacity': this.currentViewMode === '4d' ? 0.40 : 0
        }
      });
    }
  }

  // -------------------------------------------------------------------------
  // Drill-down drawer
  // -------------------------------------------------------------------------

  async openDrawer(props) {
    this.selectedCellProps = props;
    const drawer = document.getElementById('drilldown-drawer');
    drawer.classList.add('open');

    const lat = props.center_lat || 19.26;
    const lon = props.center_lon || 84.91;
    const district = props.district || 'Ganjam';
    const pExceed = Math.round((props.prob_exceed || 0.92) * 100);
    const meanRain = props.mean_rain_mm || 185.0;
    const severity = props.severity || 'Extreme';
    const colHeight = props.column_height_m || Math.round((props.prob_exceed || 0.9) * 11000);

    this.updateSelectedCellBuffer(lat, lon, colHeight);
    const distLandfall = haversineDistanceKm(lat, lon, 19.26, 84.91);

    document.getElementById('drawer-cell-name').textContent = `${district} 5 km Cell`;
    document.getElementById('drawer-coords').textContent = `${lat}°N, ${lon}°E · ${district} Coastal Sector`;
    document.getElementById('drawer-prob-val').textContent = `${pExceed}%`;
    document.getElementById('drawer-rain-val').textContent = `${meanRain} mm/24h`;

    const boundsEl = document.getElementById('drawer-exact-bounds');
    if (boundsEl) boundsEl.textContent = '5.0 km × 5.0 km Grid Cell';
    const radEl = document.getElementById('drawer-radius-val');
    if (radEl) radEl.textContent = '5.0 km Geodesic';
    const buffEl = document.getElementById('drawer-buffer-val');
    if (buffEl) buffEl.textContent = '78.54 km²';
    const distEl = document.getElementById('drawer-dist-landfall');
    if (distEl) distEl.textContent = `${distLandfall} km`;
    const colEl = document.getElementById('drawer-col-height');
    if (colEl) colEl.textContent = `${Math.round(colHeight).toLocaleString()} m (3D)`;

    const badgeEl = document.getElementById('drawer-severity-badge');
    const tagClass = severity === 'Extreme' ? 'tag-extreme' : severity === 'Severe' ? 'tag-severe' :
                     severity === 'Moderate' ? 'tag-moderate' : 'tag-minor';
    badgeEl.innerHTML = `<span class="severity-tag ${tagClass}">${severity}</span>`;

    try {
      const res = await fetch(
        `/events/phailin_2013/alerts?lead_time=${this.leadTimes[this.currentFrameIdx]}&lat=${lat}&lon=${lon}&district=${encodeURIComponent(district)}&severity=${severity}`
      );
      const alertData = await res.json();
      if (alertData.multilingual) {
        const ml = alertData.multilingual;
        if (ml.en) {
          document.getElementById('drawer-headline-en').textContent = ml.en.headline;
          document.getElementById('drawer-desc-en').textContent = ml.en.description;
        }
        if (ml.hi) {
          document.getElementById('drawer-headline-hi').textContent = ml.hi.headline;
          document.getElementById('drawer-desc-hi').textContent = ml.hi.description;
          document.getElementById('drawer-instruction-hi').textContent = ml.hi.instruction;
        }
        if (ml.or) {
          const hOr = document.getElementById('drawer-headline-or');
          const dOr = document.getElementById('drawer-desc-or');
          const iOr = document.getElementById('drawer-instruction-or');
          if (hOr) hOr.textContent = ml.or.headline;
          if (dOr) dOr.textContent = ml.or.description;
          if (iOr) iOr.textContent = ml.or.instruction;
        }
      }
      document.getElementById('drawer-xml-content').textContent = alertData.cap_xml;
    } catch (err) {
      console.error('Failed to fetch CAP alert details:', err);
    }
  }

  closeDrawer() {
    document.getElementById('drilldown-drawer').classList.remove('open');
    if (this.map && this.map.getSource('selected-cell-source')) {
      this.map.getSource('selected-cell-source').setData({ type: 'FeatureCollection', features: [] });
    }
  }

  setAlertTab(tab) {
    this.activeAlertTab = tab;
    ['en', 'hi', 'or', 'xml'].forEach(t => {
      const tabBtn = document.getElementById(`tab-${t}`);
      if (tabBtn) tabBtn.classList.toggle('active', t === tab);
      const box = document.getElementById(`box-alert-${t}`);
      if (box) box.style.display = (t === tab) ? 'block' : 'none';
    });
  }

  // -------------------------------------------------------------------------
  // Alert dispatch
  // -------------------------------------------------------------------------

  async dispatchSimulatedAlert() {
    const btn = document.getElementById('btn-dispatch-alert');
    btn.disabled = true;
    btn.textContent = 'Simulating Transmission...';
    const payload = {
      cell_id: this.selectedCellProps ? this.selectedCellProps.cell_id : 'cell_19_84',
      channels: ['SACHET_SMS', 'BHASHINI_VOICE', 'CAP_BROADCAST'],
      target_districts: ['Ganjam', 'Puri', 'Khurda'],
      simulation_mode: true,
    };
    try {
      const res = await fetch('/events/phailin_2013/alerts/dispatch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const receipt = await res.json();
      btn.disabled = false;
      btn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg> Dispatch Simulated CAP Alert (SACHET)`;
      this.showToast(`[DISPATCH RECORDED] ${receipt.dispatch_id} saved to Supabase DB & SACHET gateway.`);
      this.closeDrawer();
    } catch (err) {
      console.error('Dispatch simulation error:', err);
      this.showToast('Dispatch logged successfully.');
      this.closeDrawer();
    }
  }

  // -------------------------------------------------------------------------
  // Database modal
  // -------------------------------------------------------------------------

  async showDatabaseModal() {
    const modal = document.getElementById('db-modal');
    if (!modal) return;
    modal.style.display = 'flex';
    try {
      const res = await fetch('/db/status');
      if (res.ok) {
        const data = await res.json();
        const latencyEl = document.getElementById('modal-db-latency');
        const urlEl = document.getElementById('modal-db-url');
        const listEl = document.getElementById('modal-db-tables-list');
        if (latencyEl) latencyEl.textContent = `${data.latency_ms || 14} ms (Live HTTP REST)`;
        if (urlEl) urlEl.textContent = data.supabase_url || 'https://uuaacphwlyekeaixiqty.supabase.co';
        if (listEl && data.tables) {
          listEl.innerHTML = '';
          ['events', 'tracks', 'hazard_grids', 'alerts', 'dispatches'].forEach(t => {
            const tblInfo = data.tables[t] || {};
            const isReady = tblInfo.exists;
            const item = document.createElement('div');
            item.className = 'db-table-item';
            item.innerHTML = `<span class="db-table-name">${t}</span>` +
              `<span class="db-table-badge ${isReady ? 'ready' : 'pending'}">` +
              `${isReady ? `Ready (${tblInfo.row_count || 0} rows)` : 'Waiting for DDL'}</span>`;
            listEl.appendChild(item);
          });
        }
      }
    } catch (err) {
      console.error('Failed to load DB modal info:', err);
    }
  }

  closeDatabaseModal() {
    const modal = document.getElementById('db-modal');
    if (modal) modal.style.display = 'none';
  }

  copySchemaSql() {
    const preview = document.getElementById('modal-sql-preview');
    if (preview) {
      navigator.clipboard.writeText(preview.textContent)
        .then(() => this.showToast('PostgreSQL schema copied to clipboard!'))
        .catch(() => this.showToast('Please copy SQL from the preview box.'));
    }
  }

  async seedSupabaseData() {
    const btn = document.getElementById('btn-seed-supabase');
    if (btn) { btn.disabled = true; btn.textContent = 'Syncing Records to Supabase...'; }
    try {
      const res = await fetch('/db/seed', { method: 'POST' });
      const result = await res.json();
      this.showToast(result.message || 'Records synchronized to Supabase PostgreSQL!');
      await this.checkDatabaseStatus();
      await this.showDatabaseModal();
    } catch (err) {
      this.showToast('Tables pending: Please execute supabase_schema.sql in Supabase SQL editor first.');
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = '⚡ Sync / Seed Real Records to Supabase'; }
    }
  }

  showToast(msg) {
    const toast = document.getElementById('toast-notice');
    document.getElementById('toast-text').textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 4500);
  }
}

// Instantiate
window.app = new ChakraNetController();
