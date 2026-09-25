const { chromium } = require('playwright');
const fs = require('fs');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1366, height: 768 } });

  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);

  // Collect text from all visible and hidden elements
  const elements = await page.evaluate(() => {
    const results = [];
    const walk = (node) => {
      if (node.nodeType === Node.TEXT_NODE) {
        const text = node.textContent.trim();
        if (text.length > 0) {
          results.push({
            tag: node.parentElement ? node.parentElement.tagName : 'TEXT',
            id: node.parentElement ? node.parentElement.id : '',
            className: node.parentElement ? node.parentElement.className : '',
            text: text
          });
        }
      } else if (node.nodeType === Node.ELEMENT_NODE) {
        // Also check attributes like title, placeholder, aria-label
        ['title', 'placeholder', 'aria-label'].forEach(attr => {
          if (node.hasAttribute(attr)) {
            results.push({
              tag: node.tagName,
              id: node.id,
              className: node.className,
              attribute: attr,
              text: node.getAttribute(attr)
            });
          }
        });
        node.childNodes.forEach(walk);
      }
    };
    walk(document.body);
    return results;
  });

  console.log(`Found ${elements.length} text nodes/attributes.`);
  fs.writeFileSync('tests/extracted_texts.json', JSON.stringify(elements, null, 2));

  // Print all distinct texts
  const uniqueTexts = Array.from(new Set(elements.map(e => e.text)));
  console.log('\n--- Unique UI Texts ---');
  uniqueTexts.forEach(t => console.log(t));

  await browser.close();
})();
