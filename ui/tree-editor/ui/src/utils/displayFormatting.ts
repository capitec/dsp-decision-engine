/**
 * XSS-safe template formatting utilities for node display
 * Safely formats templates with field values without allowing script injection
 */

/**
 * Escapes HTML characters to prevent XSS
 */
const escapeHtml = (str: string | number | boolean): string => {
  if (str == null) return '';
  const value = String(str);
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');
};

/**
 * Validates that a template string only contains allowed field references
 * Prevents injection of arbitrary code or scripts
 */
const validateTemplate = (template: string, allowedFields: string[]): boolean => {
  if (!template) return true;
  
  // Extract all field references from the template
  const fieldMatches = template.match(/\{(\w+)\}/g) || [];
  const referencedFields = fieldMatches.map(match => match.slice(1, -1));
  
  // Check that all referenced fields are in the allowed list
  return referencedFields.every(field => allowedFields.includes(field));
};

/**
 * Safely formats a template string with field values
 * Escapes all values to prevent XSS and validates field references
 */
export const formatDisplayTemplate = (
  template: string,
  data: Record<string, any>,
  allowedFields?: string[]
): string => {
  if (!template) return '';
  
  // Validate template if allowed fields are provided
  if (allowedFields && !validateTemplate(template, allowedFields)) {
    console.warn('Invalid template: contains unauthorized field references');
    return '[Invalid template]';
  }
  
  // Replace field references with escaped values
  return template.replace(/\{(\w+)\}/g, (match, fieldName) => {
    const value = data[fieldName];
    
    // Handle different value types
    if (value === null || value === undefined) {
      return '[empty]';
    }
    
    // Handle arrays
    if (Array.isArray(value)) {
      if (value.length === 0) {
        return '[empty]';
      }
      // Join array elements, limiting length
      const joinedArray = value.map(v => String(v)).join(', ');
      return escapeHtml(joinedArray.length > 50 ? `${joinedArray.substring(0, 47)}...` : joinedArray);
    }

    // For objects (like record types), try to use toString or JSON
    if (typeof value === 'object') {
      // If it's a record with display properties, try to format nicely
      if (value.id !== undefined) {
        return `Record ${value.id}`;
      }
      // Fallback to JSON representation (escaped)
      return escapeHtml(JSON.stringify(value));
    }
    
    // Escape scalar values
    return escapeHtml(value);
  });
};

/**
 * Gets a default display format based on available fields
 * Creates a sensible default when no custom format is specified
 */
export const getDefaultDisplayFormat = (fields: { field_name: string }[]): string => {
  if (fields.length === 0) return '';
  if (fields.length === 1) return `{${fields[0].field_name}}`;
  
  // For multiple fields, use a comma-separated format
  return fields.map(f => `{${f.field_name}}`).join(', ');
};

/**
 * Truncates display text to a maximum length for node display
 * Adds ellipsis if truncated
 */
export const truncateDisplay = (text: string, maxLength = 20): string => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
};