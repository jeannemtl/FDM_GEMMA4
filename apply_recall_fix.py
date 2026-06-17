#!/usr/bin/env python3
"""Replace per_channel_recall in fdm_gemma4_ksweep.py with the validated-style metric."""
import re, sys

TARGET = "fdm_gemma4_ksweep.py"
NEW_FUNC = open("recall_fix.py").read()

src = open(TARGET).read()

# find 'def per_channel_recall(' and replace through the line before the next top-level 'def ' or 'def gate'
start = src.index("def per_channel_recall(")
# end at the next "\ndef gate(" (the function that follows it in the file)
end = src.index("\ndef gate(", start)
patched = src[:start] + NEW_FUNC.strip() + "\n\n" + src[end+1:]
open(TARGET, "w").write(patched)
print("patched per_channel_recall; new length", len(patched))
