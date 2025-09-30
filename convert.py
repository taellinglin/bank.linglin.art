import cairosvg
import os, re

for fname in os.listdir("./"):
    if re.match(r"banknote_\d+\.svg", fname):
        outname = fname.replace(".svg", ".png")
        cairosvg.svg2png(url=fname, write_to=outname, output_width=2048, output_height=1024)
        print(f"[+] Converted {fname} -> {outname}")
