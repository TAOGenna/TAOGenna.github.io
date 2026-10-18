"""Peru's departments as simplified SVG paths for the Yape tile.

Source: Natural Earth 1:10m admin-1 (public domain):
  curl -L -o ne_admin1.geojson https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson
Run: NE_ADMIN1=ne_admin1.geojson python scripts/tile-data/peru.py   (needs shapely)
"""
import json, math, os

from shapely.geometry import shape

src = os.environ["NE_ADMIN1"]
feats = [f for f in json.load(open(src))["features"] if f["properties"].get("admin") == "Peru"]
geoms = [(f["properties"]["name"], shape(f["geometry"])) for f in feats]

lon0 = min(g.bounds[0] for _, g in geoms); lat0 = min(g.bounds[1] for _, g in geoms)
lon1 = max(g.bounds[2] for _, g in geoms); lat1 = max(g.bounds[3] for _, g in geoms)
H = 100.0
k = H / (lat1 - lat0)
c = math.cos(math.radians((lat0 + lat1) / 2))
X = lambda lon: round((lon - lon0) * c * k, 1)
Y = lambda lat: round((lat1 - lat) * k, 1)

def path(geom):
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    out = []
    for p in polys:
        if p.area < 0.003:  # drop specks (tiny islands)
            continue
        ring = list(p.exterior.coords)
        out.append("M" + "L".join(f"{X(x)} {Y(y)}" for x, y in ring) + "Z")
    return "".join(out)

regions = []
for name, g in geoms:
    s = g.simplify(0.03, preserve_topology=True)
    cx, cy = g.representative_point().x, g.representative_point().y
    regions.append({"id": name, "d": path(s), "cx": X(cx), "cy": Y(cy)})

out = os.path.join(os.path.dirname(__file__), "../../src/data/tiles/peru.json")
json.dump({"w": round((lon1 - lon0) * c * k, 1), "h": H, "regions": regions}, open(out, "w"), separators=(",", ":"))
print(len(regions), "regions,", round(os.path.getsize(out) / 1024, 1), "KB, viewBox", round((lon1 - lon0) * c * k, 1), "x", H)
