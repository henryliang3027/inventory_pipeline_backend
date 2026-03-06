# clip_test

Builds a CLIP-based image embedding database for drink recognition using CLIP ViT-B/32 and `sentence-transformers`.

## How it works

1. Reads cropped drink images from a `my_crops/` directory, organized by drink name (one folder per category)
2. Encodes each image into a 512-dimensional embedding using CLIP ViT-B/32
3. Averages the embeddings for each category to produce a single representative feature vector
4. Saves the resulting database as `drink_db.pt` for downstream similarity search or classification

## Requirements

- Python 3.10+
- `torch`
- `sentence-transformers`
- `Pillow`
- `numpy`

Install dependencies:

```bash
pip install torch sentence-transformers Pillow numpy
```

## Usage

Organize your drink images like this:

```
my_crops/
  cola/
    img1.jpg
    img2.jpg
  juice/
    img1.jpg
    img2.png
  ...
```

Then run:

```bash
python create_clip.py
```

The output `drink_db.pt` contains a dictionary with:
- `names`: list of drink category names
- `embeddings`: tensor of shape `(num_classes, 512)`
