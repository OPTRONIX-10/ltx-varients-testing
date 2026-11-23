# Inputs

Put your own files here. Everything in this folder except this README is git-ignored, so your photos never get committed.

```
inputs/
├── subjects/
│   ├── alice/
│   │   ├── front.png      # required: full-body photo, facing the camera
│   │   ├── back.png       # required: full-body photo, same outfit, from behind
│   │   └── prompt.txt     # optional: overrides the default generic prompt
│   └── bob/
│       ├── front.jpg
│       └── back.jpg
└── motion/
    └── reference.mp4      # optional: a clip of someone turning 360°, for the motion pipeline
```

## Subjects (required)

- One folder per person. The folder name is used in result filenames.
- `front` and `back` can be `.png`, `.jpg`, `.jpeg` or `.webp`.
- Photos should be **full body (head to shoes)**, standing upright and roughly centred. A plain background works best, and the background is kept in the video.
- Any size or aspect ratio works: photos are scaled to 1280 px tall and padded or cropped to the target width.
- Without `prompt.txt`, every subject uses one generic try-on prompt (see `GENERIC_PROMPT` in `src/ltx_testing/common.py`). The photos carry the person and the outfit, so you don't need to describe them.
- Two subjects × two runs each are enough for stable timings. More subjects mean longer (more expensive) runs.

## Motion reference (optional)

Only needed for the motion-template pipeline (`motion.py` + `comfy_bench.py::motion`):

- One clip (`.mp4`, `.mov`, `.webm` or `.mkv`) of a single person turning a full 360° on the spot, with a static camera, centred, on a plain background. Any length and frame rate; it is retimed to 121 frames at 24 fps.
- Black pillar- or letterbox bars are cropped away automatically.
- Use footage you have the rights to.
