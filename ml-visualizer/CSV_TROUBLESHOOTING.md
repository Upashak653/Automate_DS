# CSV File Troubleshooting

## Common CSV Errors

### Error: "Expected X fields, saw Y"

**Cause**: Your CSV has inconsistent number of columns across rows.

**Solution**: The backend now automatically skips bad rows. If you see this error:
1. The file will still be processed (bad rows skipped)
2. Check your CSV for rows with extra commas
3. Ensure all rows have the same number of columns

### Error: "UnicodeDecodeError"

**Cause**: File encoding issue.

**Solution**:
1. Save CSV as UTF-8 encoding
2. In Excel: Save As → CSV UTF-8
3. Backend will try latin-1 encoding automatically

### Error: "File too large"

**Cause**: File exceeds size limits.

**Solution**:
1. Sample your data in Python/Excel
2. Remove unnecessary columns
3. Backend handles up to 100MB

## Best Practices

1. **Clean Headers**: No special characters in column names
2. **Consistent Columns**: All rows should have same number of fields
3. **UTF-8 Encoding**: Save with UTF-8 encoding
4. **No Empty Rows**: Remove empty rows at the end
5. **Proper Quoting**: Use quotes for text with commas

## Quick Fixes

### In Python:
```python
import pandas as pd
df = pd.read_csv('your_file.csv', on_bad_lines='skip')
df.to_csv('cleaned_file.csv', index=False)
```

### In Excel:
1. Open CSV
2. Remove empty rows
3. Save As → CSV UTF-8

The backend now handles most CSV issues automatically!
