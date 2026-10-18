"""Render two real pages of the bachelor thesis for the Thesis tile's book spread.

Source: public/Files/bachelor_thesis.pdf, printed pages 48-49 = PDF pages 56-57 (Figure 4.6 + eq. 4.41; eqs 4.44-4.46).
Run: python scripts/tile-data/thesis_pages.py   (needs pdftoppm from poppler + pillow)
White paper is multiplied by a warm paper tone so the pages sit on the dark surface like real sheets.
"""
import os, subprocess, tempfile

from PIL import Image, ImageChops

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
PDF = os.path.join(ROOT, "public/Files/bachelor_thesis.pdf")
OUT = os.path.join(ROOT, "public/projects/tiles")
PAPER = (246, 243, 235)

with tempfile.TemporaryDirectory() as d:
    subprocess.run(["pdftoppm", "-f", "56", "-l", "57", "-r", "150", "-png", PDF, os.path.join(d, "p")], check=True)
    for pdf_page, name in ((56, "thesis_p48.jpg"), (57, "thesis_p49.jpg")):
        im = Image.open(os.path.join(d, f"p-{pdf_page}.png")).convert("RGB")
        im = ImageChops.multiply(im, Image.new("RGB", im.size, PAPER))
        im.save(os.path.join(OUT, name), quality=82, optimize=True)
        print(name, im.size, round(os.path.getsize(os.path.join(OUT, name)) / 1024), "KB")
