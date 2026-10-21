"""Export a stopped RL run without resetting its initial model or bounded target.

The destination must expose storage_root at the same absolute path (a symlink
to its persistent project directory is sufficient). Source/runtime are prepared
separately. No credentials, runtime libraries, or unrelated checkpoints are copied.
Uses only the standard library so it can run on the storage access host.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def metadata(path, rules, hints):
    data = json.loads(Path(path).with_suffix('.json').read_text())
    if (data['model_sha256'] != digest(path) or data['rules'] != rules
            or data['hints'] != hints):
        raise ValueError('checkpoint integrity or rule/feature contract differs')
    return data


def export_run(run, destination, workers):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    storage = run.parent
    config = json.loads((run / 'config.json').read_text())
    if type(workers) is not int or not 1 <= workers <= config['n_envs']:
        raise ValueError('workers must be in 1..n_envs')
    if destination.exists():
        raise ValueError('archive already exists')
    with (run / 'output/.experiment.lock').open('a') as lock:
        # A running learner must be stopped before producing the final handoff.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        experiment = json.loads((run / 'output/experiment.json').read_text())
        initial = metadata(run / 'initial.zip', config['rules'], config['hints'])
        if (initial['model_sha256'] != experiment['initial_sha256']
                or initial['timesteps'] != experiment['initial_timesteps']):
            raise ValueError('initial model does not match the fixed experiment')
        candidates = []
        for path in [run / 'initial.zip', run / 'output/latest.zip', *run.glob('output/checkpoints/step_*.zip')]:
            try:
                m = metadata(path, config['rules'], config['hints'])
                candidates.append((m['timesteps'], str(path), m))
            except (OSError, ValueError, KeyError):
                continue
        steps, selected, selected_meta = max(candidates)
        if steps > experiment['target_timesteps']:
            raise ValueError('checkpoint exceeds the recorded target')
        if selected_meta['source_sha256'] != experiment['source_sha256']:
            raise ValueError('training package changed')
        files = {}

        def add(path, name=None):
            path = Path(path)
            if not path.resolve().is_relative_to(storage):
                raise ValueError('handoff file is outside experiment storage')
            name = name or str(path.relative_to(storage))
            files[name] = path

        def pair(path):
            add(path)
            add(Path(path).with_suffix('.json'))

        pair(run / 'initial.zip')
        selected = Path(selected)
        # Normalize even a valid latest into an immutable completed-update copy.
        relative = run.relative_to(storage)
        checkpoint_name = str(relative / 'output/checkpoints' / ('step_%d.zip' % steps))
        add(selected, checkpoint_name)
        add(selected.with_suffix('.json'), str(Path(checkpoint_name).with_suffix('.json')))
        best = run / 'output/best.zip'
        try:
            best_meta = metadata(best, config['rules'], config['hints'])
            if best_meta['timesteps'] <= steps:
                pair(best)
        except (OSError, ValueError, KeyError):
            pass
        pool = json.loads((run / 'output/selfplay/opponents.json').read_text())
        snapshots = []
        for entry in pool['snapshots']:
            path = Path(entry['path'])
            m = metadata(path, config['rules'], config['hints'])
            if m['model_sha256'] != entry['sha256'] or m['timesteps'] != entry['timesteps']:
                raise ValueError('self-play manifest differs from snapshot')
            if m['timesteps'] <= steps:
                pair(path)
                snapshots.append(entry)
        if not snapshots:
            raise ValueError('no compatible historical opponents')
        add(run / 'run_cpu.sh')
        config['workers'] = workers
        original_experiment = json.loads(json.dumps(experiment))
        experiment['config']['workers'] = workers
        generated = {
            str(relative / 'config.json'): config,
            str(relative / 'output/experiment.json'): experiment,
            str(relative / 'experiment-rjob.json'): original_experiment,
            str(relative / 'output/selfplay/opponents.json'): {'version': 1, 'snapshots': snapshots},
        }
        report = {'version': 1, 'storage_root': str(storage), 'run_relative': str(relative),
            'initial_timesteps': initial['timesteps'], 'resume_timesteps': steps,
            'optimizer_epochs': selected_meta['optimizer_updates'],
            'target_timesteps': experiment['target_timesteps'],
            'remaining_timesteps': experiment['target_timesteps'] - steps,
            'initial_sha256': initial['model_sha256'], 'resume_sha256': selected_meta['model_sha256'],
            'source_sha256': experiment['source_sha256'], 'workers': workers,
            'snapshot_count': len(snapshots), 'files_sha256': {n: digest(p) for n, p in files.items()}}
        generated[str(relative / 'sandbox-handoff.json')] = report
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        try:
            with tarfile.open(temporary, 'w') as archive:
                for name, path in files.items():
                    archive.add(path, arcname=name, recursive=False)
                for name, value in generated.items():
                    content = (json.dumps(value, indent=2) + '\n').encode()
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = len(content), 0o644
                    archive.addfile(info, io.BytesIO(content))
                log = run / 'output/metrics.jsonl'
                if log.exists():
                    valid = []
                    for line in log.read_text().splitlines():
                        try:
                            event = json.loads(line)
                            if event.get('timesteps', 0) <= steps:
                                valid.append(json.dumps(event))
                        except ValueError:
                            continue
                    content = ('\n'.join(valid) + '\n').encode()
                    info = tarfile.TarInfo(str(relative / 'output/metrics.jsonl'))
                    info.size, info.mode = len(content), 0o644
                    archive.addfile(info, io.BytesIO(content))
                for name in ('source', 'runtime'):
                    info = tarfile.TarInfo(str(relative / name))
                    info.type, info.linkname = tarfile.SYMTYPE, '../' + name
                    archive.addfile(info)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return dict(report, archive=str(destination), archive_sha256=digest(destination),
                    archive_bytes=destination.stat().st_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(export_run(args.run, args.output, args.workers), indent=2))
