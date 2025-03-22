// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Document Management Workflow', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the application
    await page.goto('http://localhost:3000');
  });

  test('Should upload a document and view it in the list', async ({ page }) => {
    // Check that we're on the document list page
    await expect(page.locator('h1')).toContainText('Documents');
    
    // Click the upload button
    await page.click('button:has-text("Upload Document")');
    
    // Expect upload modal to appear
    await expect(page.locator('.upload-modal')).toBeVisible();
    
    // Upload a file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'test-document.pdf',
      mimeType: 'application/pdf',
      buffer: Buffer.from('Test PDF content'),
    });
    
    // Click the submit button
    await page.click('button:has-text("Submit")');
    
    // Wait for upload to complete and modal to close
    await expect(page.locator('.upload-modal')).toBeHidden();
    
    // Verify the document appears in the list
    await expect(page.locator('.document-list')).toContainText('test-document.pdf');
  });

  test('Should search documents and view results', async ({ page }) => {
    // Enter a search query
    await page.fill('input[placeholder="Search documents..."]', 'test');
    await page.press('input[placeholder="Search documents..."]', 'Enter');
    
    // Wait for search results
    await expect(page.locator('.search-results')).toBeVisible();
    
    // Verify search results contain relevant information
    await expect(page.locator('.search-results')).toContainText('test-document.pdf');
    
    // Click on a search result
    await page.click('.search-result:first-child');
    
    // Verify document details are displayed
    await expect(page.locator('.document-details')).toBeVisible();
    await expect(page.locator('.document-details')).toContainText('test-document.pdf');
  });

  test('Should delete a document', async ({ page }) => {
    // Find the document in the list
    const documentRow = page.locator('.document-item:has-text("test-document.pdf")');
    
    // Click the delete button for this document
    await documentRow.locator('button.delete-button').click();
    
    // Confirm deletion in the modal
    await page.click('button:has-text("Confirm")');
    
    // Verify the document is removed from the list
    await expect(page.locator('.document-list')).not.toContainText('test-document.pdf');
  });
}); 