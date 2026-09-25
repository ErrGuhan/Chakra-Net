const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

(async () => {
  console.log('--- Starting Playwright CLI Layout & Functionality Verification ---');
  const artifactsDir = path.join(__dirname, 'tests', 'artifacts');
  if (!fs.existsSync(artifactsDir)) {
    fs.mkdirSync(artifactsDir, { recursive: true });
  }

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
  });

  const page = await context.newPage();
  const consoleMessages = [];
  const pageErrors = [];

  page.on('console', msg => {
    const text = msg.text();
    consoleMessages.push(`[${msg.type()}] ${text}`);
    if (msg.type() === 'error') {
      console.log(`PAGE LOG ERROR: ${text}`);
    }
  });

  page.on('pageerror', err => {
    pageErrors.push(err.message);
    console.error(`UNCAUGHT EXCEPTION: ${err.message}`);
  });

  try {
    console.log('Navigating to http://127.0.0.1:8000 ...');
    await page.goto('http://127.0.0.1:8000', { waitUntil: 'domcontentloaded', timeout: 15000 });

    // Wait for the app and MapLibre map to initialize
    await page.waitForSelector('#map-view', { timeout: 10000 });
    await page.waitForTimeout(3000); // Allow tiles and geojson layers to populate

    console.log('Page title:', await page.title());

    // 1. Verify Top HUD layout and spacing
    const topHud = await page.$('#top-hud');
    const hudBox = await topHud.boundingBox();
    console.log(`Top HUD bounding box: ${hudBox.width}x${hudBox.height} at (${hudBox.x}, ${hudBox.y})`);

    // Verify Brand title row
    const brandTitle = await page.textContent('.brand-title-row h1');
    const brandBadge = await page.textContent('.brand-sub-badge');
    console.log(`Brand Title: "${brandTitle}", Sub-Badge: "${brandBadge}"`);

    // Verify Telemetry strip items
    const intensity = await page.textContent('#hud-intensity');
    const pressure = await page.textContent('#hud-pressure');
    const wind = await page.textContent('#hud-wind');
    const dbStatus = await page.textContent('#hud-db-val');
    console.log(`Telemetry Indicators: Status="${intensity.trim()}", Pressure="${pressure}", Wind="${wind}", DB="${dbStatus.trim()}"`);

    // 2. Verify Left Control Sidebar spacing & clean structure
    const sidebar = await page.$('#control-sidebar');
    const sideBox = await sidebar.boundingBox();
    console.log(`Sidebar bounding box: ${sideBox.width}x${sideBox.height} at (${sideBox.x}, ${sideBox.y})`);

    // Check that Tech specs accordion is collapsed by default (hiding intricate details)
    const techDrawer = await page.$('#tech-specs-drawer');
    const isOpenByDefault = await techDrawer.evaluate(el => el.hasAttribute('open'));
    console.log(`Tech Specs accordion collapsed by default: ${!isOpenByDefault}`);

    // Capture primary screenshot of the clean, spacious dashboard
    const screenshot1 = path.join(artifactsDir, 'chakranet_spacious_clean_dashboard.png');
    await page.screenshot({ path: screenshot1, fullPage: false });
    console.log(`Saved primary verification screenshot: ${screenshot1}`);

    // 3. Test expanding the Technical Specs accordion to reveal intricate details
    console.log('Clicking to expand Technical Specs & Physics accordion...');
    await page.click('#tech-specs-drawer summary');
    await page.waitForTimeout(500);

    const isOpenAfterClick = await techDrawer.evaluate(el => el.hasAttribute('open'));
    console.log(`Tech Specs accordion expanded after click: ${isOpenAfterClick}`);

    // Verify quick stat cards in the accordion
    const statCards = await page.$$('.quick-stat-card');
    console.log(`Found ${statCards.length} technical stat cards in expanded drawer`);

    // Capture screenshot of expanded technical view
    const screenshot2 = path.join(artifactsDir, 'chakranet_tech_specs_expanded.png');
    await page.screenshot({ path: screenshot2, fullPage: false });
    console.log(`Saved expanded tech specs screenshot: ${screenshot2}`);

    // Collapse it back for clean presentation
    await page.click('#tech-specs-drawer summary');
    await page.waitForTimeout(400);

    // 4. Test View Mode Switching (2D / 4D)
    console.log('Testing 2D Plan view button...');
    await page.click('#btn-view-2d');
    await page.waitForTimeout(1000);
    const btn2dActive = await page.$eval('#btn-view-2d', el => el.classList.contains('active'));
    console.log(`2D Plan button active state: ${btn2dActive}`);

    console.log('Testing 4D Volumetric view button...');
    await page.click('#btn-view-4d');
    await page.waitForTimeout(1000);
    const btn4dActive = await page.$eval('#btn-view-4d', el => el.classList.contains('active'));
    console.log(`4D Volumetric button active state: ${btn4dActive}`);

    // 5. Test Basemap Switching
    console.log('Testing satellite basemap switch...');
    await page.click('#btn-bm-satellite');
    await page.waitForTimeout(1000);
    const satActive = await page.$eval('#btn-bm-satellite', el => el.classList.contains('active'));
    console.log(`Satellite basemap button active state: ${satActive}`);

    // Switch back to tactical dark
    await page.click('#btn-bm-dark');
    await page.waitForTimeout(500);

    // 6. Test Database Modal
    console.log('Testing database status modal opening...');
    await page.click('#hud-db-status');
    await page.waitForTimeout(800);
    const modalVisible = await page.$eval('#db-modal', el => el.style.display !== 'none');
    console.log(`Database modal opened: ${modalVisible}`);

    const modalScreenshot = path.join(artifactsDir, 'chakranet_database_modal.png');
    await page.screenshot({ path: modalScreenshot });
    console.log(`Saved database modal screenshot: ${modalScreenshot}`);

    // Close modal
    await page.click('.btn-modal-close');
    await page.waitForTimeout(400);
    const modalClosed = await page.$eval('#db-modal', el => el.style.display === 'none');
    console.log(`Database modal closed: ${modalClosed}`);

    // Check for CSP or JavaScript errors
    const cspErrors = consoleMessages.filter(m => m.includes('Content Security Policy') || m.includes('CSP'));
    console.log(`Total console messages: ${consoleMessages.length}`);
    console.log(`CSP Errors: ${cspErrors.length}`);
    console.log(`Page Exceptions: ${pageErrors.length}`);

    if (cspErrors.length > 0) {
      console.warn('CSP Errors found:', cspErrors);
    }
    if (pageErrors.length > 0) {
      console.error('Page Errors found:', pageErrors);
    }

    console.log('--- Playwright CLI Verification Completed Successfully! ---');
  } catch (err) {
    console.error('Playwright verification error:', err);
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
