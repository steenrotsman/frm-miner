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
                    rec = {}
                    for f in frame.fields:
                        # Don't keep coordinates for privacy
                        if f.name in ['position_lat', 'position_long']:
                            continue

                        # JSON can't hold data with timestamp type; convert to str
                        if f.name == 'timestamp':
                            rec[f.name] = str(f.value)
                        else:
                            rec[f.name] = f.value
                    record.append(rec)
            records.append(record)

    return list(filter(lambda rec: len(rec) > MIN_ACTIVITY_LEN, records))


def write_records(records):
    """Write records to JSON files."""
    for record in records:
        with open(join(JSON_DIR, record[0]['timestamp']) + '.json', 'w') as fp:
            json.dump(record, fp, indent=4)


if __name__ == '__main__':
    main()
