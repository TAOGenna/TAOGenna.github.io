"""Prepare the Majorana tile's two photos (paper first page, blackboard).

Run: python scripts/tile-data/majorana_images.py <paper_screenshot.png> <blackboard.jpg>   (needs pillow)
 - paper: flattened onto paper-white, kept at full width.
 - blackboard: cropped to the board (drops the desk and wall), grayscale, contrast-stretched.
"""
import os, sys

from PIL import Image, ImageOps

paper_src, board_src = sys.argv[1:3]
PUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../public/projects/tiles")

im = Image.open(paper_src).convert("RGBA")
bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
bg.alpha_composite(im)
bg.convert("RGB").save(PUB + "/majorana_paper.jpg", quality=88)

b = ImageOps.exif_transpose(Image.open(board_src)).convert("L")
w, h = b.size
b = b.crop((int(w * 0.03), 0, int(w * 0.945), int(h * 0.62)))  # the board only
b = ImageOps.autocontrast(b, cutoff=1)
b = b.resize((1400, round(1400 * b.height / b.width)), Image.LANCZOS)
b.convert("RGB").save(PUB + "/majorana_board.jpg", quality=84)
print("paper", Image.open(PUB + "/majorana_paper.jpg").size, "board", b.size)
