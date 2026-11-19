import unittest
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch, Mock
import dispatcher as d


class DispatcherTests(unittest.TestCase):
    def test_inventory_paginates_and_does_not_trust_gpu_default_false(self):
        from unittest.mock import MagicMock
        api = MagicMock()
        api.build_kaggle_client.return_value.__enter__.return_value.security.oauth_client.introspect_token.return_value = SimpleNamespace(active=True, username='test')
        def item(i):
            return SimpleNamespace(ref=f'test/job{i}', id=i+1, enable_gpu=False,
                                   current_version_number=0, last_run_time='2026-11-14')
        api.kernels_list_with_response.side_effect = [
            SimpleNamespace(kernels=[item(i) for i in range(100)], next_page_token=''),
            SimpleNamespace(kernels=[item(100)], next_page_token='')]
        api.kernels_status.return_value = SimpleNamespace(status=SimpleNamespace(name='QUEUED'))
        with patch.object(d,'make_api',return_value=api), patch.object(d,'quota',return_value=30), \
             patch.object(d,'read_call',side_effect=lambda fn:fn()):
            result = d.inventory('test','test-token')
        self.assertEqual(len(result['records']),101)
        self.assertEqual(result['active'],101)
        self.assertEqual(api.kernels_list_with_response.call_args_list[1].kwargs['page'],2)

    def test_confirmed_unpublished_gpu_draft_does_not_permanently_block_account(self):
        from unittest.mock import MagicMock
        import requests
        api = MagicMock()
        api.build_kaggle_client.return_value.__enter__.return_value.security.oauth_client.introspect_token.return_value = SimpleNamespace(active=True, username='test')
        api.kernels_list_with_response.return_value = SimpleNamespace(kernels=[SimpleNamespace(
            ref='test/draft', id=1, enable_gpu=False, current_version_number=0,
            last_run_time='2026-09-01')], next_page_token='')
        error = requests.HTTPError(response=SimpleNamespace(status_code=404))
        api.kernels_status.side_effect = error
        with patch.object(d,'make_api',return_value=api), patch.object(d,'quota',return_value=30), \
             patch.object(d,'read_call',side_effect=lambda fn:fn()), \
             patch.object(d,'kernel_metadata',return_value=SimpleNamespace(current_version_number=0,enable_gpu=True)):
            snapshot = d.inventory('test','test-token')
        self.assertEqual(snapshot['active'],0)
        self.assertEqual(snapshot['records']['test/draft']['status'],'NO_BATCH')

    def test_execute_reserves_before_push_and_respects_full_accounts(self):
        job = {'slug': 'mlp-fl-fedprox-s41', 'backbone': 'mlp', 'method': 'fedprox',
               'seed': 41, 'state': 'pending'}
        events = []
        api = Mock()
        response = SimpleNamespace(error='', version_number=1, ref='dangthiet/'+job['slug'],
                                   invalid_kernel_sources=[], invalid_dataset_sources=[],
                                   invalid_competition_sources=[], invalid_model_sources=[])
        def push(*args, **kwargs):
            self.assertEqual(events[-1], 'reserved')
            events.append('push')
            return response
        api.kernels_push.side_effect = push
        def inventory(owner, *args):
            return {'api':api, 'records':{}, 'active':0 if owner == 'dangthiet' else 2,
                    'free_hours':30}
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'queue.json').write_text(json.dumps({'jobs':[job]}))
            with patch.object(d, 'ROOT', Path(folder)), patch.object(d, 'checked_inventory', side_effect=inventory), \
                 patch.object(d, 'quota', return_value=30), patch.object(d, 'build', return_value=Path(folder)), \
                 patch.object(d, 'confirm_submission', return_value=1), patch('builtins.print'), \
                 patch.object(d, 'save', side_effect=lambda state, commit: events.append(state['jobs'][0]['state'])), \
                 patch.object(d.sys, 'argv', ['dispatcher.py','--execute']), \
                 patch.dict(os.environ, {'KAGGLE_TOKENS_JSON':json.dumps(dict.fromkeys(d.ACCOUNTS,'test')),
                                         'GITHUB_TOKEN':'', 'GITHUB_STEP_SUMMARY':''}):
                self.assertEqual(d.main(), 0)
        self.assertEqual(api.kernels_push.call_count, 1)
        self.assertEqual(events[-1], 'submitted')

    def test_existing_on_another_account_never_relaunches(self):
        jobs = [{'slug': 'mlp-fl-fedavg-s41', 'state': 'pending'}]
        d.reconcile(jobs, {'nguynt298/mlp-fl-fedavg-s41': {'version': 1, 'status': 'ERROR'}})
        self.assertEqual(jobs[0]['state'], 'submitted')
        self.assertEqual(jobs[0]['remote_status'], 'ERROR')

    def test_lost_response_without_visible_kernel_requires_review(self):
        jobs = [{'slug': 'mlp-fl-fedavg-s41', 'state': 'reserved'}]
        d.reconcile(jobs, {})
        self.assertEqual(jobs[0]['state'], 'review')

    def test_lost_response_with_visible_kernel_is_reconciled(self):
        jobs = [{'slug': 'mlp-fl-fedavg-s41', 'state': 'reserved'}]
        d.reconcile(jobs, {'minhiw/mlp-fl-fedavg-s41': {'version': 1, 'status': 'QUEUED'}})
        self.assertEqual(jobs[0]['state'], 'submitted')

    def test_duplicate_kernel_requires_review(self):
        jobs = [{'slug': 'x', 'state': 'pending'}]
        d.reconcile(jobs, {'a/x': {}, 'b/x': {}})
        self.assertEqual(jobs[0]['state'], 'review')

    def test_exit_zero_style_error_is_not_success(self):
        response = SimpleNamespace(error='Maximum batch GPU session count of 2 reached.', version_number=0)
        self.assertFalse(d.accepted(response, 'a/x'))

    def test_acknowledgement_rejects_invalid_inputs(self):
        response = SimpleNamespace(error='', version_number=1, ref='a/x', invalid_kernel_sources=[],
                                   invalid_dataset_sources=[], invalid_competition_sources=[], invalid_model_sources=[])
        self.assertTrue(d.accepted(response, 'a/x'))
        response.invalid_kernel_sources = ['missing-parent']
        self.assertFalse(d.accepted(response, 'a/x'))

    def test_sparse_acknowledgement_defers_identity_to_remote_verification(self):
        response = SimpleNamespace(error='', version_number=0, ref='', url='',
                                   invalid_kernel_sources=[], invalid_dataset_sources=[],
                                   invalid_competition_sources=[], invalid_model_sources=[])
        self.assertTrue(d.accepted(response,'a/x'))

    def test_remote_confirmation_rejects_wrong_source_or_identity(self):
        from unittest.mock import MagicMock
        api = MagicMock()
        meta = SimpleNamespace(ref='a/x', is_private=True, enable_gpu=True, current_version_number=1)
        nb = {'cells':[{'cell_type':'code','source':'print(1)'}]}
        result = SimpleNamespace(metadata=meta, blob=SimpleNamespace(source=json.dumps(nb)))
        api.build_kaggle_client.return_value.__enter__.return_value.kernels.kernels_api_client.get_kernel.return_value = result
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder/'kernel-metadata.json').write_text(json.dumps({'code_file':'x.ipynb'}))
            (folder/'x.ipynb').write_text(json.dumps(nb))
            with patch.object(d,'read_call',side_effect=lambda fn:fn()):
                self.assertEqual(d.confirm_submission(api,'a/x',folder),1)
                meta.ref='b/x'
                with self.assertRaises(RuntimeError): d.confirm_submission(api,'a/x',folder)
                meta.ref='a/x'
                result.blob.source=json.dumps({'cells':[{'cell_type':'code','source':'print(2)'}]})
                with self.assertRaises(RuntimeError): d.confirm_submission(api,'a/x',folder)

    def test_parent_complete_status_is_not_enough(self):
        api = Mock()
        from unittest.mock import MagicMock
        api.build_kaggle_client.return_value = MagicMock()
        inner = api.build_kaggle_client.return_value.__enter__.return_value
        inner.kernels.kernels_api_client.list_kernel_session_output.return_value = SimpleNamespace(
            files=[], next_page_token=None)
        self.assertFalse(d.parent_ready(api, 'a/sae-fl-fedavg-s41', 1,
                                       {'backbone': 'sae', 'seed': 41}))

    def test_parent_requires_round50_matching_seed_and_backbone(self):
        from unittest.mock import MagicMock
        api = MagicMock()
        inner = api.build_kaggle_client.return_value.__enter__.return_value
        inner.kernels.kernels_api_client.list_kernel_session_output.return_value = SimpleNamespace(
            files=[SimpleNamespace(file_name=name, url=name) for name in
                   ('RUN_STATUS.json','experiment_manifest.json','sae-fl-fedavg-s41/state.pt')],
            next_page_token=None)
        status = {'complete':True,'last_round':49,'all_round_metrics':True}
        manifest = {'rounds':50,'method':'fedavg','seed':41,
                    'experiment':'single-backbone-sae-baselines'}
        def get(url, **kwargs):
            return Mock(json=Mock(return_value=status if url=='RUN_STATUS.json' else manifest))
        with patch.object(d.requests, 'get', side_effect=get):
            job = {'backbone':'sae','seed':41}
            self.assertFalse(d.parent_ready(api,'a/sae-fl-fedavg-s41',1,job))
            status['last_round'] = 50
            self.assertTrue(d.parent_ready(api,'a/sae-fl-fedavg-s41',1,job))
            job['seed'] = 42
            self.assertFalse(d.parent_ready(api,'a/sae-fl-fedavg-s41',1,job))


if __name__ == '__main__':
    unittest.main()
