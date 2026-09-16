"""Record actual SQLite names and columns; do not infer new schema mappings."""
import json
from pathlib import Path
import sqlite3
import sys

def describe_schema(connection):
    names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    return {'schema': 'observed-nsys-sqlite-schema/v1', 'tables': {name: [
        {'cid': row[0], 'name': row[1], 'type': row[2], 'notnull': row[3], 'default': row[4], 'pk': row[5]}
        for row in connection.execute('PRAGMA table_info("' + name.replace('"','""') + '")')]
        for name in names}, 'column_mappings_guessed': False}

if __name__ == '__main__':
    source = Path(sys.argv[1]).resolve()
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as connection:
        print(json.dumps(describe_schema(connection), ensure_ascii=False, indent=2))
