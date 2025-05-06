import numpy as np
import matplotlib.pyplot as plt

def place_text_corner(ax, text, corner="bottom left", offset=0.03, **kwargs):
    positions = {
        "bottom left":  (offset, offset),
        "bottom right": (1 - offset, offset),
        "top left":     (offset, 1 - offset),
        "top right":    (1 - offset, 1 - offset),
    }
    x, y = positions.get(corner.lower(), (offset, offset))
    ax.text(x, y, text, transform=ax.transAxes,
            ha='left' if 'left' in corner else 'right',
            va='bottom' if 'bottom' in corner else 'top',
            **kwargs)
