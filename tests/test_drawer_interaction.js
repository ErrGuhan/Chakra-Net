const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();

  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);

  // Trigger drawer programmatically or by clicking
  await page.evaluate(() => {
    // Call openDrawer with realistic Gopalpur cell properties
    window.app.openDrawer({
      cell_id: 'cell_19_84',
      center_lat: 19.26,
      center_lon: 84.91,
      prob_exceed: 0.92,
      mean_rain_mm: 185.0,
      severity: 'Extreme',
      threshold_mm: 75.0,
      district: 'Ganjam',
      lead_time_hours: 108
    });
  });

  await page.waitForTimeout(1000);

  await page.screenshot({ path: 'tests/artifacts/test_drawer_open.png' });
  console.log('Captured tests/artifacts/test_drawer_open.png');

  // Test language tab switching
  await page.click('#tab-hi');
  await page.waitForTimeout(500);
  await page.screenshot({ path: 'tests/artifacts/test_drawer_hindi.png' });
  console.log('Captured tests/artifacts/test_drawer_hindi.png');

  await page.click('#tab-xml');
  await page.waitForTimeout(500);
  await page.screenshot({ path: 'tests/artifacts/test_drawer_xml.png' });
  console.log('Captured tests/artifacts/test_drawer_xml.png');

  // Test dispatch button
  await page.click('#btn-dispatch-alert');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'tests/artifacts/test_drawer_dispatched.png' });
  console.log('Captured tests/artifacts/test_drawer_dispatched.png');

  await browser.close();
})();
