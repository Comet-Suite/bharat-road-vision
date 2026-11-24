# Attribution

Every image keeps its upstream licence. Training or redistributing this dataset means honouring each one.

| Licence | Records | Source |
|---|---|---|
| CC BY-SA 4.0 | 608 | commons |
| CC BY-SA 3.0 | 309 | commons |
| BY | 173 | openverse |
| CC BY 3.0 | 111 | commons |
| BY-SA | 97 | openverse |
| CC BY-SA 2.0 | 67 | commons |
| CC BY 4.0 | 54 | commons |
| CC0 | 48 | commons |
| CC BY 2.0 | 35 | commons |
| GODL-India | 31 | commons |
| Public domain | 23 | commons |
| BY-2.0 | 10 | openverse |
| CC BY 2.5 | 8 | commons |
| BY-SA-2.0 | 8 | openverse |
| CC BY 3.0 pl | 5 | commons |
| CC0 | 2 | openverse |
| Attribution | 1 | commons |
| CC BY-SA 2.5 in | 1 | commons |
| CC BY-SA 2.5 | 1 | commons |
| CC0-1.0 | 1 | openverse |

## Per-record credits

Each JSONL record carries `license`, `attribution`, `source_page` and `original_url`:

```python
import json
for r in map(json.loads, open("data/bharat_road_vision.jsonl")):
    print(r["attribution"], "|", r["license"], "|", r["source_page"])
```

CC BY / CC BY-SA images require credit to the named author on redistribution; CC0 / public-domain images do not.
