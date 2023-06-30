import json
from os import listdir, remove
from os.path import join

import fitdecode

FIT_DIR = 'fit'
JSON_DIR = 'bike'

# Don't count activities shorter than 5 minutes
MIN_ACTIVITY_LEN = 300


def main():
    for file in listdir(FIT_DIR):
        fn = join(JSON_DIR, file[:19]) + '.json'
        length = 0
        with open(fn, 'w') as fp:
            fp.write('[\n')
            with fitdecode.FitReader(join(FIT_DIR, file)) as fit:
                for frame in fit:
                    if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == 'record':
                        length += 1
                        rec = get_rec(frame)
                        if length > 1:
                            fp.write(',\n')
                        json.dump(rec, fp, indent=4)
            fp.write('\n]')
        if length < MIN_ACTIVITY_LEN:
            remove(fn)


def get_rec(frame):
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

    return rec


if __name__ == '__main__':
    main()
