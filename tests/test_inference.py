"""Offline tests use invented text and mocked model responses only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import dspy
import httpx
import aihub
import boundary_reassert
import dspy_prompts as prompts
import run
from evidence_span_gate import apply_evidence_span_gate
from ldl_goal_gate import apply_ldl_goal_gate

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    def test_original_source_hashes(self):
        manifest = json.loads((ROOT / 'source_manifest.json').read_text())
        for name, digest in manifest['original_sha256'].items():
            if name != 'boundary_reassert.py':
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)

    def test_no_implicit_demonstrations(self):
        self.assertEqual(boundary_reassert.load_boundary_demos(), [])
        with self.assertRaises(FileNotFoundError):
            boundary_reassert.load_boundary_demos('/nonexistent/example.json')

    def test_supplied_demonstrations(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'demo.json'
            p.write_text(json.dumps([{'text': 'Invented example.', 'corrected_boundary_labels': ['continue_current_therapy'], 'boundary_evidence': {}}]))
            examples = boundary_reassert.load_boundary_demos(p)
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].text, 'Invented example.')

    def test_router_branches(self):
        for label in ['escalation', 'no change', 'de-escalation', 'change', 'bad note']:
            with self.subTest(label=label):
                router = prompts.PromptRouter_ChainOfThought()
                router.route_logic = Mock(return_value=dspy.Prediction(llt_change=label))
                router.escalation = Mock(return_value=dspy.Prediction(llt_reasons=['ldl_above_goal']))
                router.nonescalation = Mock(return_value=dspy.Prediction(llt_reasons=['continue_current_therapy']))
                with patch.object(prompts, 'apply_boundary_reassert', side_effect=lambda x, **kw: x) as boundary, patch.object(prompts, 'apply_nonescalation_gates', side_effect=lambda x, **kw: x) as gates:
                    result = router(context='invented', text='Invented note.')
                self.assertEqual(result['llt_change'], label)
                self.assertEqual(router.escalation.call_count, int(label == 'escalation'))
                self.assertEqual(router.nonescalation.call_count, int(label in ['no change', 'de-escalation', 'change']))
                self.assertEqual(boundary.call_count, router.nonescalation.call_count)
                self.assertEqual(gates.call_count, router.nonescalation.call_count)
                if label == 'bad note':
                    self.assertEqual(result['llt_reasons'], [])

    def test_gate_order(self):
        from contextlib import ExitStack
        names = ['apply_ldl_goal_gate', 'apply_evidence_span_gate', 'apply_defer_recall_add', 'apply_defer_labs_verifier', 'apply_other_suppression_gate']
        seen = []
        def gate(name):
            def apply(value, **kw):
                seen.append(name)
                return value
            return apply
        with ExitStack() as stack:
            for name in names:
                stack.enter_context(patch.object(prompts, name, side_effect=gate(name)))
            out = prompts.apply_nonescalation_gates({'reason_evidence': {}, 'llt_reasons': []}, text='Invented.')
        self.assertEqual(seen, names)
        self.assertNotIn('reason_evidence', out)

    def test_filters(self):
        out = apply_ldl_goal_gate({'llt_reasons': ['ldl_at_goal'], 'ldl_goal_status': 'above'})
        self.assertEqual(out['llt_reasons'], ['continue_current_therapy'])
        out = apply_evidence_span_gate({'llt_reasons': ['side_effects'], 'reason_evidence': {'side_effects': ['unrelated quote']}}, text='Invented note.')
        self.assertEqual(out['llt_reasons'], ['continue_current_therapy'])
        self.assertNotIn('reason_evidence', out)

    def test_transport(self):
        seen = []
        transport = aihub.HubTransport('synthetic-test-key')
        transport.transport.close()
        def handle(request):
            seen.append(request)
            return httpx.Response(200, json={})
        transport.transport = httpx.MockTransport(handle)
        with httpx.Client(transport=transport) as client:
            client.post(aihub.GATEWAY + '/chat/completions', json={'model': 'gpt-5'}, headers={'Authorization': 'Bearer placeholder'})
        self.assertEqual(seen[0].url.path, '/azure-openai/deployments/gpt-5/chat/completions')
        self.assertNotIn('authorization', seen[0].headers)
        self.assertEqual(seen[0].headers['api-key'], 'synthetic-test-key')

    def test_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            note, output = Path(tmp) / 'note.txt', Path(tmp) / 'output.json'
            note.write_text('Invented note.')
            prediction = {'decision': {'llt_change': 'no change', 'llt_reasons': ['continue_current_therapy']}, 'ldl_target': {'ldl_target_discussed': 'not discussed', 'ldl_target_value': 'None'}}
            with patch.object(run, 'configure_model'), patch.object(run, 'extract', return_value=(prediction, 0)):
                self.assertEqual(run.main(['--note', str(note), '--output', str(output)]), 0)
                self.assertEqual(json.loads(output.read_text())['configuration']['version'], 'v83')
                original = output.read_bytes()
                with self.assertRaises(SystemExit):
                    run.main(['--note', str(note), '--output', str(output)])
                self.assertEqual(output.read_bytes(), original)

    def test_exception_redaction(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            note, output = Path(tmp) / 'note.txt', Path(tmp) / 'out.json'
            note.write_text('Invented note.')
            stream = io.StringIO()
            with patch.object(run, 'configure_model'), patch.object(run, 'extract', side_effect=RuntimeError('SYNTHETIC_SECRET_MARKER')), contextlib.redirect_stderr(stream):
                self.assertEqual(run.main(['--note', str(note), '--output', str(output)]), 1)
            self.assertNotIn('SYNTHETIC_SECRET_MARKER', stream.getvalue())
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
