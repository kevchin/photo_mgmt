# Image Orientation Fix Strategy

## Problem Summary

You have photos in your database showing with wrong orientation. Key constraints:
- Many images already captioned (don't want to lose this work)
- Images organized in YYYY/MM/DD directory structure (must preserve)
- Can't easily determine which need rotation without AI/LLM detection
- Don't want duplicate files with new dates

## Recommended Solution: Database-Driven Rotation

### Overview
Instead of creating rotated copies of images, store the **rotation angle** in the database and apply rotation at **display time**. This approach:

✅ No duplicate files needed  
✅ Preserves original file dates and directory structure  
✅ Maintains all existing captions and metadata  
✅ Works incrementally  
✅ Minimal storage overhead  

### Implementation Steps

#### Step 1: Add Database Column

Run this once to add an `orientation_correction` column:

```bash
cd /workspace/image_utils
python add_orientation_column.py "postgresql://user:password@localhost:5432/your_database"
```

This adds:
- `orientation_correction INTEGER DEFAULT 0` - stores rotation angle (0, 90, 180, 270)
- Index for quick lookup of images needing correction

#### Step 2: Detect Which Images Need Rotation

Run the detection script (can be done in batches):

```bash
# Check 100 images (dry run first)
python detect_orientation_issues.py "postgresql://..." --batch-size 100 --dry-run

# Actually update the database
python detect_orientation_issues.py "postgresql://..." --batch-size 100

# Run periodically to process more images
python detect_orientation_issues.py "postgresql://..." --batch-size 100
```

The script:
- Checks EXIF orientation tags (fast, reliable for camera photos)
- Can use AI analysis for images without EXIF (optional enhancement)
- Updates the database with required rotation angle
- Marks images as checked so you don't reprocess them

#### Step 3: Update Your Display Application

Modify your image display code to check the `orientation_correction` field:

```sql
-- Get images that need rotation
SELECT id, file_path, orientation_correction 
FROM images 
WHERE orientation_correction != 0;
```

Then apply rotation when displaying:

**Python/PIL Example:**
```python
from PIL import Image

def display_image(file_path: str, rotation_angle: int):
    with Image.open(file_path) as img:
        if rotation_angle != 0:
            # PIL rotates counter-clockwise, so negate
            img = img.rotate(-rotation_angle, expand=True)
        img.show()  # or save/display as needed
```

**Web/JavaScript Example:**
```javascript
// CSS transform for display
<img src="photo.jpg" style="transform: rotate(90deg);" />

// Or Canvas API for more control
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
ctx.translate(canvas.width/2, canvas.height/2);
ctx.rotate(angle * Math.PI / 180);
ctx.drawImage(img, -img.width/2, -img.height/2);
```

### Alternative: Physical File Rotation

If you MUST have physically rotated files (e.g., for external tools), use this workflow:

```bash
# Create a script to rotate files in place
python rotate_files_in_place.py "postgresql://..." --backup
```

This would:
1. Query images with `orientation_correction != 0`
2. Rotate the actual file
3. Remove EXIF orientation tag (already applied)
4. Keep same filename and path
5. Optionally backup originals

⚠️ **Risks:**
- File modification date changes
- Potential data loss if something goes wrong
- Breaks any external references to specific file hashes

### Handling Future Uploads

Update your ingestion pipeline to check orientation during import:

```python
# In ingest_images.py or similar
from image_orientation import correct_image_orientation

# During ingestion
was_corrected, corrected_path, reason = correct_image_orientation(
    file_path, overwrite=False
)

if was_corrected:
    # Store the rotation angle in database
    orientation_angle = extract_rotation_from_reason(reason)
    # Continue processing with corrected_path
    # But store ORIGINAL file_path in database
```

### Migration Path for Existing Images

For your existing database:

1. **Immediate fix**: Add column + run detection (Steps 1-2 above)
2. **Display fix**: Update your viewer to respect `orientation_correction`
3. **Optional cleanup**: Later, if desired, physically rotate files in batch

### Why This Approach?

| Concern | Solution |
|---------|----------|
| Duplicate files | ❌ No duplicates - rotation is metadata |
| Date preservation | ✅ Original file dates unchanged |
| Directory structure | ✅ Files stay in YYYY/MM/DD |
| Caption preservation | ✅ Captions linked to DB record, not file |
| Incremental processing | ✅ Process in batches, track progress |
| External tool compatibility | ⚠️ Some tools may need physical rotation |

### Advanced: AI-Based Detection Enhancement

The current `detect_orientation_issues.py` uses EXIF tags. For images without EXIF or to verify EXIF accuracy, you can enhance it to use the Florence-2 model from `auto_rotate.py`:

```python
# In analyze_image_orientation(), add:
if exif_rotation == 0 or verify_with_ai:
    # Use Florence-2 to analyze scene
    rotation_angle, reason, confidence = analyze_orientation(image_path)
```

This is slower but catches issues EXIF misses (e.g., scanner uploads, edited images).

## Summary

**Don't create duplicate rotated files.** Instead:

1. Add `orientation_correction` column to database
2. Run detection to populate it
3. Apply rotation at display time

This preserves everything you've built while fixing the orientation issue cleanly.
