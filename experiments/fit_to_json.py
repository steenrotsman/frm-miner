import json
from os import listdir
from os.path import join

import fitdecode

FIT_DIR = 'fit'
JSON_DIR = 'bike'

# Don't count activities shorter than 5 minutes
MIN_ACTIVITY_LEN = 300


def main():
    records = get_records(FIT_DIR)
    records = parse_records(records)
    write_records(records)


def get_records(directory):
    """Read records from FIT files into a list of dictionaries."""
    records = []
    for file in listdir(directory):
        record = []

        with fitdecode.FitReader(join(directory, file)) as fit:
            for frame in fit:
                # Only keep frame data, no personal or metadata
                if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == 'record':
                    # JSON can't hold data with timestamp type
                    record.append({f.name: f.value if f.name != 'timestamp' else str(f.value) for f in frame.fields})

            records.append(record)

    return list(filter(lambda rec: len(rec) > MIN_ACTIVITY_LEN, records))


def parse_records(records):
    """Convert timestamps to strings and coordinates to lat-longs."""
    for i, record in enumerate(records):
        for j, rec in enumerate(record):
            # Convert timestamp to string
            records[i][j]['timestamp'] = str(rec['timestamp'])

            # Convert lat and long to float
            for field in ['position_lat', 'position_long']:
                # https://gis.stackexchange.com/a/371667
                records[i][j][field] = rec[field] / 11930465

    return records


def write_records(records):
    """Write records to JSON files."""
    for record in records:
        with open(join(JSON_DIR, record[0]['timestamp']) + '.json', 'w') as fp:
            json.dump(record, fp, indent=4)


if __name__ == '__main__':
    main()
