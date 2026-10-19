"""Topic counts + headline numbers for the Library tile (Competitive programming block).

Parses the root README of competitive-programming (read-only): the HTML <p> of topics with their
problem counts, "2,496 problems in 98 topics", the template count and the blog table.
Run: python scripts/tile-data/cp_library.py   (stdlib only)
"""
import json, os, re

README = os.path.expanduser("~/Desktop/kenyi/projects/competitive-programming/README.md")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/data/tiles/cp_library.json")

s = open(README, encoding="utf-8").read()
topics = [
    {"name": n.replace("&nbsp;", " "), "count": int(c)}
    for n, c in re.findall(r'<a href="Collection_of_problems/README\.md#[^"]*">([^<]+)</a>&nbsp;<sub>(\d+)</sub>', s)
]
problems, n_topics = re.search(r"([\d,]+) problems in (\d+) topics", s).groups()
templates = int(re.search(r"\|\s*\[`Algorithms_templates/`\][^|]*\|\s*(\d+) C\+\+ templates", s).group(1))
blog_cats = [(n.strip(), int(c)) for n, c in re.findall(r"^\| ([A-Za-z][^|]+?) \| (\d+) \|$", s, re.M)]
blogs = sum(c for _, c in blog_cats)

out = {"problems": int(problems.replace(",", "")), "nTopics": int(n_topics), "templates": templates, "blogs": blogs, "topics": topics}
assert len(topics) == out["nTopics"], (len(topics), out["nTopics"])
assert blogs == 140 and templates == 33, (blogs, templates)
json.dump(out, open(OUT, "w"), indent=1)
print(len(topics), "topics,", out["problems"], "problems,", templates, "templates,", blogs, "blogs; top:", sorted(topics, key=lambda t: -t["count"])[:4])
