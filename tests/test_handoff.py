"""Handoff must preserve the original target and reject unsafe/live exports."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('policy_export', Path(__file__).resolve().parents[1] / 'scripts/export_policy_handoff.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HandoffTests(unittest.TestCase):
    def fixture(self, root):
        run = root / 'run'
        (run / 'output/checkpoints').mkdir(parents=True)
        (run / 'output/selfplay').mkdir()
        config = {'workers': 2, 'n_envs': 8, 'rules': {}, 'hints': True}
        (run / 'config.json').write_text(json.dumps(config))
        (run / 'run_cpu.sh').write_text('echo fixture\n')

        def model(path, steps):
            path.write_bytes(('model-%d' % steps).encode())
            m = {'timesteps': steps, 'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'rules': {}, 'hints': True, 'optimizer_updates': steps // 4, 'source_sha256': 'code'}
            path.with_suffix('.json').write_text(json.dumps(m))
            return m

        initial = model(run / 'initial.zip', 0)
        model(run / 'output/checkpoints/step_16.zip', 16)
        bad = model(run / 'output/latest.zip', 32)
        (run / 'output/latest.zip').write_bytes(b'torn')
        snapshot = run / 'output/selfplay/snapshot.zip'
        m = model(snapshot, 8)
        (run / 'output/selfplay/opponents.json').write_text(json.dumps({'version': 1,
            'snapshots': [{'path': str(snapshot), 'timesteps': 8, 'sha256': m['model_sha256']}]}))
        (run / 'output/experiment.json').write_text(json.dumps({'initial_sha256': initial['model_sha256'],
            'initial_timesteps': 0, 'target_timesteps': 64, 'config': config, 'source_sha256': 'code'}))
        return run

    def test_torn_latest_falls_back_without_resetting_target_or_pool(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.fixture(root)
            archive = root / 'handoff.tar'
            report = module.export_run(run, archive, 4)
            self.assertEqual(report['resume_timesteps'], 16)
            self.assertEqual(report['target_timesteps'], 64)
            self.assertEqual(report['remaining_timesteps'], 48)
            self.assertEqual(json.loads((run / 'config.json').read_text())['workers'], 2)
            with tarfile.open(archive) as contents:
                manifest = json.load(contents.extractfile('run/output/experiment.json'))
                self.assertEqual(manifest['initial_timesteps'], 0)
                self.assertEqual(manifest['target_timesteps'], 64)
                self.assertEqual(manifest['config']['workers'], 4)
                self.assertIn('run/output/selfplay/snapshot.zip', contents.getnames())
                self.assertNotIn('run/output/latest.zip', contents.getnames())
                self.assertTrue(all(not name.startswith('/') and '..' not in Path(name).parts
                                    for name in contents.getnames()))

    def test_active_learner_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.fixture(root)
            with (run / 'output/.experiment.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    module.export_run(run, root / 'handoff.tar', 4)
            self.assertFalse((root / 'handoff.tar').exists())


if __name__ == '__main__':
    unittest.main()
