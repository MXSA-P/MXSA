import time
from difflib import SequenceMatcher
def sim(a,b): return SequenceMatcher(None, a, b).ratio()
words = ["please"] * 100000
pattern = "hello world"
t0 = time.time()
for i in range(len(words)-2):
    sim(" ".join(words[i:i+2]), pattern)
print(time.time()-t0)
