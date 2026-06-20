import re
re.compile(r"[^\w\s'-]")
re.compile(r"(?<![\w'-])test(?![\w'-])")
re.compile(r"(?i)(?<![\w'-])(?:please|now|quickly|fast|right now|for me)\s*$")
print("All compile successfully!")
