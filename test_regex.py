import re
text = "high-speed"
pattern = "speed"
print(bool(re.search(rf"(?<![\w'-]){re.escape(pattern)}(?![\w'-])", text)))
