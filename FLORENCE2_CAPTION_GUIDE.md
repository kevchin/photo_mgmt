# Using Florence-2 for Photo Captioning in Photo Archive Ingest

This guide explains how to use Microsoft's Florence-2 model for generating captions during photo ingestion, including how to specify different levels of caption detail.

## Overview

The `photo_archive_ingest.py` tool now supports two types of local LLM backends:

1. **Florence-2** (HuggingFace) - Specialized vision-language model for detailed image captions
2. **Ollama models** (e.g., Llama 3) - General-purpose LLMs with vision capabilities

Florence-2 is recommended for photo captioning as it produces more accurate and detailed descriptions.

## Quick Start

### Basic Usage with Florence-2

```bash
python photo_archive_ingest.py \
  --dir ~/Documents/trip_photos \
  --archive trip_test \
  --llm-model microsoft/Florence-2-base
```

### Detailed Captions

```bash
python photo_archive_ingest.py \
  --dir ~/Documents/trip_photos \
  --archive trip_test \
  --llm-model microsoft/Florence-2-base \
  --caption-detail detailed
```

### Very Detailed Captions

```bash
python photo_archive_ingest.py \
  --dir ~/Documents/trip_photos \
  --archive trip_test \
  --llm-model microsoft/Florence-2-large \
  --caption-detail very_detailed
```

## Caption Detail Levels

| Level | Description | Use Case | Speed |
|-------|-------------|----------|-------|
| `basic` | Short, simple caption (1 sentence) | Quick ingestion, large archives | Fastest |
| `detailed` | More descriptive (2-3 sentences) | General purpose | Moderate |
| `very_detailed` | Comprehensive description | Small archives, high quality | Slowest |

## Model Options

### Florence-2 Models

| Model | Size | Quality | VRAM Required |
|-------|------|---------|---------------|
| `microsoft/Florence-2-base` | ~230MB | Good | ~2GB |
| `microsoft/Florence-2-large` | ~580MB | Better | ~4GB |

**Recommendation**: Start with `Florence-2-base` with `detailed` captions for most use cases.

### Ollama Models

If you prefer using Ollama (already running locally):

```bash
python photo_archive_ingest.py \
  --dir ~/Documents/photos \
  --archive my_archive \
  --llm-model llama3:8b
```

Note: Ollama models may produce less consistent captions than Florence-2.

## Configuration via YAML

You can set default models and caption detail levels in your `archives_config.yaml`:

```yaml
archives:
  - name: "Trip Photos"
    id: "trip_test"
    db_path: "postgresql://postgres:password@localhost:5432/photo_trip"
    root_dir: "~/Documents/trip_photos"
    description: "Recent trip photos with detailed Florence-2 captions"
    llm_model: "microsoft/Florence-2-base"
    caption_detail: "detailed"
    
  - name: "Quick Archive"
    id: "quick_v1"
    db_path: "postgresql://postgres:password@localhost:5432/photo_quick"
    root_dir: "~/Documents/quick_photos"
    description: "Large archive with basic captions for speed"
    llm_model: "microsoft/Florence-2-base"
    caption_detail: "basic"

settings:
  default_llm_model: "microsoft/Florence-2-base"
  default_caption_detail: "detailed"
```

Then simply run:

```bash
python photo_archive_ingest.py --dir ~/Documents/trip_photos --archive trip_test
```

The tool will automatically use the configured model and detail level.

## Command Line Override Priority

The tool uses this priority order for determining settings:

1. **Command line arguments** (highest priority)
2. **Archive-specific config** in YAML
3. **Global defaults** in YAML settings section

Example: Even if your archive is configured with `basic` captions, you can override for a specific run:

```bash
python photo_archive_ingest.py \
  --dir ~/Documents/special_photos \
  --archive trip_test \
  --caption-detail very_detailed \
  --llm-model microsoft/Florence-2-large
```

## Complete Example Workflow

### Step 1: Create Database

```bash
psql -U postgres -c "CREATE DATABASE photo_trip_archive;"
```

### Step 2: Initialize Schema

```bash
python image_database.py init \
  --db "postgresql://postgres:YOUR_PASSWORD@localhost:5432/photo_trip_archive"
```

### Step 3: Add Archive to Config

Edit `archives_config.yaml`:

```yaml
- name: "Trip Photos Detailed"
  id: "trip_detailed"
  db_path: "postgresql://postgres:YOUR_PASSWORD@localhost:5432/photo_trip_archive"
  root_dir: "~/Documents/trip_photos"
  description: "Trip photos with very detailed Florence-2 captions"
  llm_model: "microsoft/Florence-2-large"
  caption_detail: "very_detailed"
```

### Step 4: Ingest Photos

```bash
python photo_archive_ingest.py \
  --dir ~/Documents/trip_photos \
  --archive trip_detailed \
  --batch-size 20
```

### Step 5: View in Streamlit

```bash
streamlit run streamlit_app.py
```

Select "Trip Photos Detailed" from the archive dropdown.

## Performance Tips

1. **Batch Size**: Use smaller batch sizes (10-20) for large models or limited VRAM
2. **GPU**: Florence-2 automatically uses GPU if available (CUDA/MPS)
3. **First Run**: Initial model download takes time (~5 minutes for base, ~15 for large)
4. **Memory**: Close other GPU applications when processing large batches

## Troubleshooting

### Out of Memory Errors

Reduce batch size or use smaller model:
```bash
python photo_archive_ingest.py \
  --dir ~/Documents/photos \
  --archive my_archive \
  --llm-model microsoft/Florence-2-base \
  --batch-size 10
```

### Slow Processing

- Use `basic` caption detail for faster processing
- Use `Florence-2-base` instead of `large`
- Ensure GPU is being used (check for "Using GPU:" message)

### Model Download Issues

Pre-download models:
```python
from transformers import AutoProcessor, AutoModelForCausalLM
AutoProcessor.from_pretrained("microsoft/Florence-2-base")
AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base")
```

## Comparing Caption Models

To test different captioning approaches on the same photos:

```bash
# Test 1: Florence-2 basic
python photo_archive_ingest.py \
  --dir ~/Documents/test_photos \
  --archive test_florence_basic \
  --llm-model microsoft/Florence-2-base \
  --caption-detail basic

# Test 2: Florence-2 detailed
python photo_archive_ingest.py \
  --dir ~/Documents/test_photos \
  --archive test_florence_detailed \
  --llm-model microsoft/Florence-2-base \
  --caption-detail detailed

# Test 3: Llama 3
python photo_archive_ingest.py \
  --dir ~/Documents/test_photos \
  --archive test_llama3 \
  --llm-model llama3:8b
```

Then compare results in Streamlit by switching between archives.

## See Also

- `simple_caption.py` - Test caption generation on single images
- `generate_captions_local.py` - Lower-level caption generation utilities
- `QUICK_START_NEW_ARCHIVE.md` - Complete archive setup guide
