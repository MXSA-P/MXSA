import re

with open("simba/core/memory.py", "r") as f:
    content = f.read()

# Fix E302
content = content.replace('\nlogger = get_logger("simba.core.memory")\n\nclass NumpyEncoder(json.JSONEncoder):', '\nlogger = get_logger("simba.core.memory")\n\n\nclass NumpyEncoder(json.JSONEncoder):')

# Fix E501 line 278
content = content.replace('                json_str = json.dumps(self._objects, indent=2, ensure_ascii=False, cls=NumpyEncoder)', '                json_str = json.dumps(\n                    self._objects, indent=2, ensure_ascii=False, cls=NumpyEncoder\n                )')

# Fix E501 line 285
content = content.replace('                fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(self.db_path), prefix="memory_", suffix=".tmp")', '                fd, tmp_path = tempfile.mkstemp(\n                    dir=os.path.dirname(self.db_path), prefix="memory_", suffix=".tmp"\n                )')

# Fix E501 line 339
content = content.replace('                            entry["last_seen_ts"] = datetime.fromisoformat(entry["last_seen_timestamp"]).timestamp()', '                            entry["last_seen_ts"] = datetime.fromisoformat(\n                                entry["last_seen_timestamp"]\n                            ).timestamp()')

# Fix E501 line 417
content = content.replace('                        last_seen_ts = datetime.fromisoformat(entry["last_seen_timestamp"]).timestamp()', '                        last_seen_ts = datetime.fromisoformat(\n                            entry["last_seen_timestamp"]\n                        ).timestamp()')

# Fix W293 (blank lines containing whitespace)
content = re.sub(r'^[ \t]+$', '', content, flags=re.MULTILINE)

with open("simba/core/memory.py", "w") as f:
    f.write(content)
