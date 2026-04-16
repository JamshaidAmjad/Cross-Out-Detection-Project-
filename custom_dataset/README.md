# Custom Dataset

This folder holds manually created cross-out images used to supplement the main dataset.

## What to Add

Create approximately **50 images** organised into subfolders matching the 8 class names exactly:

```
custom_dataset/
├── CLEAN/          # Pages with no cross-outs
├── CROSS/          # X-shaped cross-outs
├── DIAGONAL/       # Single diagonal line cross-outs
├── DOUBLE_LINE/    # Two parallel horizontal/diagonal lines
├── SCRATCH/        # Irregular, scribble-style cross-outs
├── SINGLE_LINE/    # One horizontal line through text
├── WAVE/           # Wavy/curved line cross-outs
└── ZIG_ZAG/        # Zig-zag pattern cross-outs
```

## Guidelines

- Aim for **~6–7 images per class** (balanced distribution)
- Images should be scanned or photographed at a reasonable resolution (min 300×300 px)
- File format: `.jpg` or `.png`
- Folder names must match **exactly** (uppercase) so `ImageFolder` picks them up correctly
- Do **not** create a `MIXED` folder — it is excluded from training

## Integration

Once images are added here, point `cfg.data_dir` (in `config/config.py`) to this folder
or merge it with the main dataset before running training.
