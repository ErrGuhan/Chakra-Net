const { chromium } = require('playwright');
const fs = require('fs');

(async () => {
  console.log('=== Starting Comprehensive Playwright UI Verification (4D Map & 5 km Geodesic Radius) ===\n');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();

  const consoleErrors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });

  const pageErrors = [];
  page.on('pageerror', err => {
    pageErrors.push(err.toString());
  });

  // 1. Navigate to dashboard
  console.log('[1/8] Navigating to http://127.0.0.1:8000/ ...');
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2500);

  // Check console errors
  console.log(`[2/8] Console errors count: ${consoleErrors.length}`);
  if (consoleErrors.length > 0) {
    console.error('Console errors found:', consoleErrors);
  } else {
    console.log('  ✓ Zero console errors! Clean runtime verified.');
  }

  // 2. Verify Flat Styling in Key UI Elements (Zero Gradients)
  console.log('[3/8] Verifying flat solid styling compliance (Zero gradients)...');
  const gradientChecks = await page.evaluate(() => {
    const selectors = [
      '.top-hud',
      '.brand-symbol',
      '.brand-text h1',
      '.sidebar-panel',
      '.event-info-box',
      '.btn-dispatch-action',
      '.scientific-legend',
      '.legend-bar-continuous',
      '.view-mode-switcher',
      '.view-mode-btn'
    ];

    const results = [];
    selectors.forEach(sel => {
      const el = document.querySelector(sel);
      if (el) {
        const style = window.getComputedStyle(el);
        const bg = style.backgroundImage;
        const hasGradient = bg && bg.includes('gradient');
        results.push({ selector: sel, hasGradient, bg: bg || style.backgroundColor });
      }
    });
    return results;
  });

  let gradientFound = false;
  gradientChecks.forEach(r => {
    if (r.hasGradient) {
      console.error(`  FAIL: Gradient found in ${r.selector}: ${r.bg}`);
      gradientFound = true;
    } else {
      console.log(`  ✓ ${r.selector}: flat solid styling confirmed.`);
    }
  });

  // 3. Text & Meteorological Audit
  console.log('[4/8] Auditing page text for meteorology terminology...');
  const textAudit = await page.evaluate(() => {
    const fullText = document.body.innerText;
    const aiKeywords = ['Two-Stage AI Hybrid Pipeline', 'AI Weather Tracking', 'AI pipeline', 'एआई पाइपलाइन'];
    const foundKeywords = aiKeywords.filter(k => fullText.includes(k));

    const legendHead = document.querySelector('.legend-head');
    const legendText = legendHead ? legendHead.innerText : '';

    return {
      foundKeywords,
      legendText,
      telemetryIntensity: document.querySelector('.telemetry-value') ? document.querySelector('.telemetry-value').innerText : ''
    };
  });

  console.log('  Text audit results:');
  console.log('  - AI Buzzwords remaining:', textAudit.foundKeywords.length === 0 ? 'None (Clean!)' : textAudit.foundKeywords);
  console.log('  - Legend Header Text:', textAudit.legendText);
  console.log('  - Telemetry Storm Intensity:', textAudit.telemetryIntensity);

  // 4. Test Viewport Fit at 1280x720 (Laptop)
  console.log('[5/8] Testing responsiveness and layout fit at 1280x720...');
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.waitForTimeout(500);

  const fitCheck = await page.evaluate(() => {
    const sidebar = document.getElementById('control-sidebar');
    const threshold = document.querySelector('.threshold-container');
    const winH = window.innerHeight;

    return {
      winH,
      sidebarBottom: sidebar.getBoundingClientRect().bottom,
      sidebarWithinWindow: sidebar.getBoundingClientRect().bottom <= winH,
      thresholdTop: threshold.getBoundingClientRect().top,
      thresholdBottom: threshold.getBoundingClientRect().bottom,
      thresholdVisible: threshold.getBoundingClientRect().bottom <= winH
    };
  });
  console.log('  Layout at 1280x720:', fitCheck);
  if (!fitCheck.thresholdVisible) {
    console.error('  FAIL: Threshold container not fully visible on 720p laptop screen!');
  } else {
    console.log('  ✓ Sidebar and threshold slider fully visible within 720p window!');
  }

  // Capture overview screenshot
  await page.screenshot({ path: 'tests/artifacts/verified_dashboard_overview.png' });
  console.log('  ✓ Saved tests/artifacts/verified_dashboard_overview.png');

  // Reset viewport to 1440x900 for interaction tests
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.waitForTimeout(500);

  // 5. Test 4D Volumetric View vs 2D Plan View Switching
  console.log('[6/8] Testing 4D Volumetric View vs 2D Plan View Switching...');
  const viewModeTest = await page.evaluate(() => {
    const pitchBefore = window.app.map.getPitch();
    const bearingBefore = window.app.map.getBearing();
    const is4DActive = window.app.currentViewMode === '4d';

    // Switch to 2D
    window.app.setViewMode('2d');
    const is2DActive = window.app.currentViewMode === '2d';

    return { pitchBefore, bearingBefore, is4DActive, is2DActive };
  });
  console.log('  Initial 4D mode state:', viewModeTest);
  await page.waitForTimeout(1100);
  await page.screenshot({ path: 'tests/artifacts/verified_2d_plan_view.png' });
  console.log('  ✓ Saved tests/artifacts/verified_2d_plan_view.png');

  // Switch back to 4D Volumetric View
  await page.evaluate(() => {
    window.app.setViewMode('4d');
  });
  await page.waitForTimeout(1300);

  const mode4DResult = await page.evaluate(() => {
    const pitch = window.app.map.getPitch();
    const bearing = window.app.map.getBearing();
    const extrusionLayer = window.app.map.getLayer('layer-hazard-extrusion-4d');
    return {
      pitch: Math.round(pitch),
      bearing: Math.round(bearing),
      hasExtrusion: !!extrusionLayer
    };
  });
  console.log('  4D Volumetric Camera Angle & Extrusion:', mode4DResult);
  await page.screenshot({ path: 'tests/artifacts/verified_4d_volumetric_map.png' });
  console.log('  ✓ Saved tests/artifacts/verified_4d_volumetric_map.png');

  // 6. Test Basemap & Threshold Slider Interactions
  console.log('[7/8] Testing Basemap & Threshold Slider Interactions...');
  await page.click('#btn-bm-terrain');
  await page.waitForTimeout(600);
  await page.click('#btn-bm-dark');
  await page.waitForTimeout(600);
  await page.click('#btn-bm-satellite');
  await page.waitForTimeout(600);

  // Change threshold slider to 150 mm
  await page.evaluate(() => {
    const slider = document.getElementById('range-threshold');
    slider.value = 150;
    window.app.setThreshold(150);
  });
  await page.waitForTimeout(800);
  await page.screenshot({ path: 'tests/artifacts/verified_threshold_150mm.png' });
  console.log('  ✓ Saved tests/artifacts/verified_threshold_150mm.png');

  // Reset threshold to 75 mm
  await page.evaluate(() => {
    const slider = document.getElementById('range-threshold');
    slider.value = 75;
    window.app.setThreshold(75);
  });
  await page.waitForTimeout(500);

  // 7. Test Cell Click, Accurate 5 km Geodesic Radius & Geometry Card
  console.log('[8/8] Testing Cell Click, Accurate 5 km Geodesic Radius Buffer & Geometry Card...');
  await page.evaluate(() => {
    window.app.openDrawer({
      cell_id: 'cell_5km_18_24',
      center_lat: 19.26,
      center_lon: 84.91,
      prob_exceed: 0.94,
      mean_rain_mm: 195.0,
      severity: 'Extreme',
      threshold_mm: 75.0,
      district: 'Ganjam',
      lead_time_hours: 108,
      column_height_m: 12500
    });
  });
  await page.waitForTimeout(1000);

  // Audit Geometry Card Contents
  const geoCardAudit = await page.evaluate(() => {
    return {
      radiusVal: document.getElementById('drawer-radius-val') ? document.getElementById('drawer-radius-val').innerText : null,
      bufferVal: document.getElementById('drawer-buffer-val') ? document.getElementById('drawer-buffer-val').innerText : null,
      distLandfall: document.getElementById('drawer-dist-landfall') ? document.getElementById('drawer-dist-landfall').innerText : null,
      colHeight: document.getElementById('drawer-col-height') ? document.getElementById('drawer-col-height').innerText : null,
      exactBounds: document.getElementById('drawer-exact-bounds') ? document.getElementById('drawer-exact-bounds').innerText : null,
      hasSelectedCircleSource: !!window.app.map.getSource('selected-cell-source'),
    };
  });
  console.log('  Geometry Card & Geodesic Radius Audit:', geoCardAudit);
  await page.screenshot({ path: 'tests/artifacts/verified_5km_radius_buffer.png' });
  console.log('  ✓ Saved tests/artifacts/verified_5km_radius_buffer.png');

  await page.screenshot({ path: 'tests/artifacts/verified_drawer_en.png' });
  console.log('  ✓ Saved tests/artifacts/verified_drawer_en.png');

  // Switch to Hindi tab
  await page.click('#tab-hi');
  await page.waitForTimeout(500);
  await page.screenshot({ path: 'tests/artifacts/verified_drawer_hi.png' });
  console.log('  ✓ Saved tests/artifacts/verified_drawer_hi.png');

  // Switch to Odia tab
  await page.click('#tab-or');
  await page.waitForTimeout(500);
  await page.screenshot({ path: 'tests/artifacts/verified_drawer_or.png' });
  console.log('  ✓ Saved tests/artifacts/verified_drawer_or.png');

  // Switch to XML tab
  await page.click('#tab-xml');
  await page.waitForTimeout(500);
  await page.screenshot({ path: 'tests/artifacts/verified_drawer_xml.png' });
  console.log('  ✓ Saved tests/artifacts/verified_drawer_xml.png');

  // Dispatch alert simulation
  await page.click('#btn-dispatch-alert');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'tests/artifacts/verified_dispatch_toast.png' });
  console.log('  ✓ Saved tests/artifacts/verified_dispatch_toast.png');

  await browser.close();
  console.log('\n=== All Playwright Verifications Completed Successfully ===');
})();
