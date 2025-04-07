# Video Generator Demo Script

This script demonstrates the usage of the `VideoGenerator` class for creating videos from images with multiple sequential captions and background audio. It uses introspection to automatically adapt to any changes in the `VideoGenerator.create_video_from_slides` method parameters.

## Prerequisites

- Python 3.9+
- ImageMagick installed and properly configured
- Required Python packages (install via pip):
  - moviepy
  - Pillow
  - numpy

## Asset Directory Structure

The script expects a directory containing numbered subdirectories, each containing an image and its caption files:

```
assets_dir/
    ├── 1/                  # First slide
    │   ├── image.jpg      # Image for first slide (must be named 'image')
    │   ├── 1.txt          # First caption for the image
    │   ├── 2.txt          # Second caption for the image
    │   └── 3.txt          # Third caption for the image
    ├── 2/                  # Second slide
    │   ├── image.png      # Image for second slide
    │   ├── 1.txt          # First caption
    │   └── 2.txt          # Second caption
    └── ...
```

Requirements:
- Each slide must be in its own numbered directory
- Each slide directory must contain exactly one image file named 'image' (can be .jpg, .jpeg, or .png)
- Caption files must be numbered .txt files (1.txt, 2.txt, etc.)
- Caption files must not be empty
- Slides and captions are processed in numerical order

## Usage

### Basic Usage

```bash
python video_generator_demo.py --assets-dir /path/to/assets
```

For example:

```bash
python -m app.scripts.tasks.video_generator_demo --assets-dir app/assets/d/video_generator/creation_1/
```

This will:
1. Load all slides (image + captions) from the asset directory
2. Create a video with default settings where:
   - Each slide shows for 5 seconds by default
   - Captions appear sequentially within each slide's duration
   - Each caption's duration is evenly distributed within the slide's duration
3. Save the output as `video_YYYYMMDD_HHMMSS.mp4` in the project's `output` directory

### Advanced Usage

```bash
python video_generator_demo.py \
    --assets-dir /path/to/assets \
    --duration-per-slide 7 \
    --font-size 40 \
    --font-color yellow \
    --audio-volume 0.7
```

### Arguments

The script automatically detects available parameters from the `VideoGenerator.create_video_from_slides` method. Core arguments are:

| Argument | Description | Required |
|----------|-------------|----------|
| `--assets-dir` | Directory containing slide subdirectories | Yes |

Additional arguments are dynamically generated based on the `VideoGenerator` class implementation. To see all available arguments, run:

```bash
python video_generator_demo.py --help
```

## Output

All generated videos are saved in the project's `output` directory with timestamped filenames:
- Location: `PROJECT_ROOT/output/`
- Filename format: `video_YYYYMMDD_HHMMSS.mp4`
- The `output` directory is automatically created if it doesn't exist
- The `output` directory is git-ignored

## Features

- **Dynamic Parameter Handling**: The script automatically adapts to changes in the `VideoGenerator` class without requiring updates
- **Type Safety**: Arguments are automatically typed according to the `VideoGenerator` method's type hints
- **Default Values**: Uses the same default values as defined in the `VideoGenerator` class
- **Self-Documenting**: Help text and available arguments are generated automatically from the source
- **Paired Assets**: Ensures each image has a corresponding caption file
- **Standardized Output**: All videos are saved in a consistent location with timestamped names

## Error Handling

The script includes robust error handling for common issues:
- Missing image or caption files
- Empty caption files
- Invalid image files
- Mismatched image-caption pairs

## Maintainability

This script is designed to be maintainable and resilient to changes:

- If new parameters are added to `VideoGenerator.create_video_from_slides`, they automatically become available as command-line arguments
- If parameter types change in the `VideoGenerator` class, the script automatically adapts
- If default values change in the `VideoGenerator` class, they are automatically reflected in the script
- The help text automatically updates to show all available options 