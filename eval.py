import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import constants
from dataset.base_dataset_test import BaseDataset_test
from models.evaluator import evaluator


def run_evaluation(evaluator_ode, checkpoint, dataset_names, batch_size, num_workers):
    log_dir = checkpoint.parent
    if log_dir.name == 'checkpoints':
        log_dir = log_dir.parent

    for name in dataset_names:
        print(f'Evaluating {name}')
        dataset = BaseDataset_test(name, config.DATASET_FOLDERS_TEST, config.hist_length)
        if not len(dataset):
            raise ValueError(f'The {name} test index contains no windows')
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        branches = ('data', 'physics', 'fusion')
        mpjpe_batches = {branch: [] for branch in branches}
        accel_batches = {branch: [] for branch in branches}
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Eval'):
                batch = {key: value.to(evaluator_ode.device) for key, value in batch.items()}
                _, metrics = evaluator_ode(batch, 'torch', dataset_name=name)
                for branch, mpjpe, accel in zip(branches, metrics[:3], metrics[3:6]):
                    mpjpe_batches[branch].append(mpjpe)
                    accel_batches[branch].append(accel)

        mpjpe = {branch: np.concatenate(batches) for branch, batches in mpjpe_batches.items()}
        future = {branch: values[:, config.hist_length:].mean(axis=0) for branch, values in mpjpe.items()}
        print(f'*** Final results for {name} ***')
        for frames in (2, 4, 8, 10, 14, 18, 22, 25):
            print(f'Evaluation at {round(frames * constants.dt * 1000)} ms')
            for branch in branches:
                print(f'MPJPE_{branch}: {future[branch][frames - 1]:.1f} mm')
                if name == 'H36M':
                    print(f'MPJPE_{branch} (P2): {future[branch][:frames].mean():.1f} mm')
        for branch in branches:
            # The first acceleration value is centered at the second input frame.
            accel = np.concatenate(accel_batches[branch])[:, config.hist_length - 1:]
            print(f'ACCEL_{branch}: {accel.mean():.1f}')

        output = log_dir / 'eval' / name
        output.mkdir(parents=True, exist_ok=True)
        np.savez(output / 'results.npz', **{f'mpjpe_{branch}': values for branch, values in mpjpe.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to network checkpoint')
    parser.add_argument('--dataset', choices=config.DATASET_FOLDERS_TEST, help='Evaluate one dataset')
    parser.add_argument('--batch-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    dataset_names = (args.dataset,) if args.dataset else config.DATASET_FOLDERS_TEST
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    for name in dataset_names:
        test_index = Path(config.DATASET_FOLDERS_TEST[name])
        if not test_index.is_file():
            parser.error(f'Missing {name} test index: {test_index}')

    evaluator_ode = evaluator(checkpoint_path=args.checkpoint)
    run_evaluation(evaluator_ode, args.checkpoint, dataset_names, args.batch_size, args.num_workers)

