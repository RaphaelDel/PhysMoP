import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import config
from dataset.motion_preprocessing import smpl_to_generalized_coordinates


SUBJECTS = {'test': (5,), 'train': (1, 6, 7, 8, 9, 11)}
RAW_DIR = Path('dataset/dataset_raw/H36M')
OUTPUT_DIR = Path('dataset/data_processed')


def preprocess(raw_dir, output_dir, split):
    sources = [
        (subject, raw_dir / f'Human36M_subject{subject}_SMPL_NeuralAnnot.json')
        for subject in SUBJECTS[split]
    ]
    for _, source in sources:
        if not source.is_file():
            raise FileNotFoundError(f'Missing Human3.6M annotations: {source}')

    length = config.total_length
    stride = length if split == 'test' else 5
    index_path = output_dir / f'h36m_{split}_{length}.pkl'
    windows_dir = output_dir / index_path.stem
    windows_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for subject, source in sources:
        print(f'Processing S{subject} ({split})', flush=True)
        with source.open(encoding='utf-8') as file:
            actions = json.load(file)
        for action, subactions in sorted(actions.items(), key=lambda item: int(item[0])):
            for subaction, frames in sorted(subactions.items(), key=lambda item: int(item[0])):
                frames = {int(key): frame for key, frame in frames.items() if frame is not None}
                # The source is 50 Hz. Split at missing 25 Hz samples to avoid spanning gaps.
                frame_ids = np.array(sorted(idx for idx in frames if idx % 2 == 0), dtype=int)
                if len(frame_ids) < length + 30:
                    continue
                breaks = np.flatnonzero(np.diff(frame_ids) != 2) + 1
                shape = np.asarray([frame['shape'][:10] for frame in frames.values()]).mean(axis=0)
                for segment in np.split(frame_ids, breaks):
                    if len(segment) < length + 30:
                        continue
                    poses = np.asarray([frames[idx]['pose'] for idx in segment], dtype=np.float32)
                    translations = np.asarray([frames[idx]['trans'] for idx in segment], dtype=np.float32)
                    q = smpl_to_generalized_coordinates(poses, translations)
                    # PhysMoP uses model ID 0 for these fits, as specified by the author.
                    annotations = np.zeros((len(q), 74), dtype=np.float32)
                    annotations[:, :63] = q
                    annotations[:, 63:73] = shape
                    for start in range(0, len(q) - length + 1, stride):
                        first_frame = segment[start + 15]
                        path = windows_dir / f'S{subject}_a{action}_s{subaction}_f{first_frame}'
                        np.save(path, annotations[start:start + length])
                        paths.append(str(path))

    if not paths:
        raise ValueError(f'No usable {length}-frame Human3.6M windows for {split}')
    with index_path.open('wb') as file:
        pickle.dump(paths, file)
    print(f'Saved {len(paths)} windows to {index_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Human3.6M NeuralAnnot SMPL fits for PhysMoP.')
    parser.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--split', choices=('test', 'train', 'both'), default='test')
    args = parser.parse_args()
    for split in SUBJECTS if args.split == 'both' else (args.split,):
        preprocess(args.raw_dir, args.output_dir, split)
