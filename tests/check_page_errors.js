const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 }
  });
  const page = await context.newPage();

  const consoleLogs = [];
  page.on('console', msg => {
    consoleLogs.push({ type: msg.type(), text: msg.text() });
  });

  const pageErrors = [];
  page.on('pageerror', err => {
    pageErrors.push(err.toString());
  });

  const failedRequests = [];
  page.on('requestfailed', req => {
    failedRequests.push({ url: req.url(), failure: req.failure() });
  });

  console.log('Navigating to http://127.0.0.1:8000/ ...');
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' });

  // Wait 2 seconds for MapLibre tiles and async data
  await page.waitForTimeout(2000);

  console.log('\n--- Console Logs ---');
  consoleLogs.forEach(l => console.log(`[${l.type}] ${l.text}`));

  console.log('\n--- Page Errors ---');
  pageErrors.forEach(e => console.log(e));

  console.log('\n--- Failed Requests ---');
  failedRequests.forEach(r => console.log(`${r.url}: ${JSON.stringify(r.failure)}`));

  // Extract all text content
  const pageText = await page.evaluate(() => {
    return document.body.innerText;
  });

  console.log('\n--- Page Text Sample (First 500 chars) ---');
  console.log(pageText.slice(0, 500));

  // Check bounding box of sidebar vs window height
  const layoutMetrics = await page.evaluate(() => {
    const sidebar = document.getElementById('control-sidebar');
    const hud = document.getElementById('top-hud');
    const bottomDock = document.getElementById('bottom-dock');
    const mapLegend = document.getElementById('map-legend');
    const thresholdContainer = document.querySelector('.threshold-container');

    return {
      windowHeight: window.innerHeight,
      windowWidth: window.innerWidth,
      hudBottom: hud ? hud.getBoundingClientRect().bottom : null,
      sidebarRect: sidebar ? {
        top: sidebar.getBoundingClientRect().top,
        bottom: sidebar.getBoundingClientRect().bottom,
        height: sidebar.getBoundingClientRect().height,
        scrollHeight: sidebar.scrollHeight,
        clientHeight: sidebar.clientHeight,
        isScrollable: sidebar.scrollHeight > sidebar.clientHeight
      } : null,
      thresholdVisible: thresholdContainer ? {
        top: thresholdContainer.getBoundingClientRect().top,
        bottom: thresholdContainer.getBoundingClientRect().bottom,
        inViewport: thresholdContainer.getBoundingClientRect().bottom <= window.innerHeight
      } : null
    };
  });

  console.log('\n--- Layout Metrics ---');
  console.log(JSON.stringify(layoutMetrics, null, 2));

  await browser.close();
})();
