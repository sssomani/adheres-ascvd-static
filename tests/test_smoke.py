"""Exercise the real CLI, DSPy, and HTTP stack against a local synthetic API."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]


def response(**fields):
    return '\n\n'.join(f'[[ ## {k} ## ]]\n{json.dumps(v) if isinstance(v, (list, dict)) else v}' for k, v in fields.items()) + '\n\n[[ ## completed ## ]]'


class SmokeTests(unittest.TestCase):
    def test_compatible_cli_http_pipeline(self):
        self.check_pipeline("compatible")

    def test_aihub_cli_http_pipeline(self):
        self.check_pipeline("aihub")

    def check_pipeline(self, provider):
        replies = [
            response(reasoning='Synthetic classification.', llt_change='no change'),
            response(reasoning='Synthetic rationale.', recent_ldl_value='92', ldl_goal_status='above', llt_reasons=['patient_preference'], other_reasons={}, reason_evidence={'patient_preference': ['declines adding ezetimibe']}),
            response(add_defer_after_labs='no'),
            response(reasoning='Synthetic target.', ldl_target_discussed='discussed', ldl_target_value='<70', other_target=''),
        ]
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append((self.path, body))
                content = replies[min(len(calls) - 1, len(replies) - 1)]
                payload = json.dumps({'id': 'synthetic', 'object': 'chat.completion', 'created': 1, 'model': 'gpt-5', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'prediction.json'
                env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower() and k not in ('PYTHONPATH', 'SECURE_GPT_KEY', 'MODEL_PROVIDER', 'MODEL_NAME', 'MODEL_BASE_URL')}
                env.update(SECURE_GPT_KEY='synthetic-test-key', OPENAI_API_KEY='synthetic-test-key', LITELLM_LOCAL_MODEL_COST_MAP='True', DSPY_CACHEDIR=str(Path(tmp) / 'cache'))
                command = [sys.executable, str(ROOT / 'src' / 'run.py')]
                if provider == 'aihub':
                    # Redirect only the public gateway address; use the actual HubTransport.
                    bootstrap = f'import sys; sys.path.insert(0, {str(ROOT / "src")!r}); import aihub, run; aihub.GATEWAY = "http://127.0.0.1:{server.server_port}/azure-openai"; sys.exit(run.main())'
                    command = [sys.executable, '-c', bootstrap]
                command += ['--provider', provider, '--model', 'openai/gpt-5', '--base-url', f'http://127.0.0.1:{server.server_port}/v1', '--note', str(ROOT / 'examples/note.txt'), '--output', str(out)]
                result = subprocess.run(command, cwd=tmp, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(out.read_text())
                self.assertEqual(data['decision']['llt_change'], 'no change')
                self.assertEqual(data['decision']['llt_reasons'], ['patient_preference'])
                self.assertEqual(data['ldl_target']['ldl_target_discussed'], 'discussed')
                self.assertEqual(data['ldl_target']['ldl_target_value'], '<70')
                self.assertNotIn('reason_evidence', data['decision'])
                self.assertEqual(data['configuration']['demonstration_count'], 0)
                self.assertEqual(len(calls), 4)
                expected = '/v1/chat/completions' if provider == 'compatible' else '/azure-openai/deployments/gpt-5/chat/completions?api-version=2025-04-01-preview'
                self.assertTrue(all(path == expected for path, _ in calls), [path for path, _ in calls])
                self.assertTrue(all(body['model'] == 'gpt-5' for _, body in calls))
                self.assertNotIn('declines adding ezetimibe', result.stdout + result.stderr)
                self.assertEqual(out.stat().st_mode & 0o777, 0o600)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
