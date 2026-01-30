# Utility: Print Unicode codepoints for each character in a file
import sys

if len(sys.argv) != 2:
    print("Usage: python print_unicode_codepoints.py <filename>")
    sys.exit(1)

filename = sys.argv[1]
with open(filename, "r", encoding="utf-8") as f:
    text = f.read()

for i, line in enumerate(text.splitlines(), 1):
    print(f"Line {i}:")
    for c in line:
        print(f"  U+{ord(c):04X} '{c}'")
    print()
