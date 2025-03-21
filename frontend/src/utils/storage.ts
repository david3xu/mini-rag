/**
 * Utility functions for local storage
 */

/**
 * Save data to local storage
 * @param key Storage key
 * @param data Data to store
 */
export function saveToStorage<T>(key: string, data: T): void {
  try {
    const serializedData = JSON.stringify(data);
    localStorage.setItem(key, serializedData);
  } catch (error) {
    console.error('Error saving to localStorage:', error);
  }
}

/**
 * Load data from local storage
 * @param key Storage key
 * @param defaultValue Default value if not found
 */
export function loadFromStorage<T>(key: string, defaultValue: T): T {
  try {
    const serializedData = localStorage.getItem(key);
    if (serializedData === null) {
      return defaultValue;
    }
    return JSON.parse(serializedData) as T;
  } catch (error) {
    console.error('Error loading from localStorage:', error);
    return defaultValue;
  }
}

/**
 * Remove data from local storage
 * @param key Storage key
 */
export function removeFromStorage(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch (error) {
    console.error('Error removing from localStorage:', error);
  }
}

/**
 * Clear all data from local storage
 */
export function clearStorage(): void {
  try {
    localStorage.clear();
  } catch (error) {
    console.error('Error clearing localStorage:', error);
  }
}