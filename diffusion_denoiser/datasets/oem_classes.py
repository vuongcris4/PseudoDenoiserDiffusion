"""OEM dataset class definitions and official color palette.

8 land-cover classes + nodata.  Raw label values 1-8 map to indices 0-7.
Raw value 0 (nodata / unlabeled) maps to IGNORE_INDEX (255).

Reference colour table (HEX → RGB):
    #800000  Bareland         (128,   0,   0)
    #00FF24  Rangeland        (  0, 255,  36)
    #949494  Developed space  (148, 148, 148)
    #FFFFFF  Road             (255, 255, 255)
    #226126  Tree             ( 34,  97,  38)
    #0045FF  Water            (  0,  69, 255)
    #4BB549  Agriculture land ( 75, 181,  73)
    #DE1F07  Building         (222,  31,   7)
"""

import numpy as np

NUM_CLASSES = 8
IGNORE_INDEX = 255

CLASS_NAMES = [
    'Bareland',       # 0
    'Rangeland',      # 1
    'Developed',      # 2
    'Road',           # 3
    'Tree',           # 4
    'Water',          # 5
    'Agriculture',    # 6
    'Building',       # 7
]

# Official OEM colour palette (RGB uint8)
CLASS_COLORS = np.array([
    [128,   0,   0],  # 0 Bareland     #800000
    [  0, 255,  36],  # 1 Rangeland    #00FF24
    [148, 148, 148],  # 2 Developed    #949494
    [255, 255, 255],  # 3 Road         #FFFFFF
    [ 34,  97,  38],  # 4 Tree         #226126
    [  0,  69, 255],  # 5 Water        #0045FF
    [ 75, 181,  73],  # 6 Agriculture  #4BB549
    [222,  31,   7],  # 7 Building     #DE1F07
], dtype=np.uint8)

# Colour for nodata / ignore pixels in visualizations
NODATA_COLOR = np.array([40, 40, 40], dtype=np.uint8)  # dark grey


def colorize_mask(mask: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Convert class-index mask (H, W) → RGB (H, W, 3).

    Pixels with value == IGNORE_INDEX are rendered in NODATA_COLOR.
    """
    h, w = mask.shape
    rgb = np.full((h, w, 3), NODATA_COLOR, dtype=np.uint8)
    for c in range(min(num_classes, len(CLASS_COLORS))):
        rgb[mask == c] = CLASS_COLORS[c]
    return rgb
