#!/usr/bin/env python3
"""Run the final v83 extraction model on one UTF-8 clinical note."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys

# Use the packaged model metadata without an unrelated network fetch.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def configure_model(provider, model, base_url):
    import dspy
    import litellm

    # Disable persistent caches and routine SDK logs containing request content.
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    litellm.suppress_debug_info = True
    for name in ('litellm', 'LiteLLM', 'httpx', 'openai', 'dspy'):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    if provider == 'aihub':
        from aihub import GATEWAY, get_openai_client
        key = os.environ.get('SECURE_GPT_KEY')
        if not key:
            raise ValueError('SECURE_GPT_KEY is required for AI Hub')
        client = get_openai_client(key)
        litellm.client_session = client._client
        endpoint = GATEWAY
    else:
        key = os.environ.get('OPENAI_API_KEY')
        if not key or not base_url:
            raise ValueError('OPENAI_API_KEY and --base-url are required for a compatible endpoint')
        endpoint = base_url
    lm = dspy.LM(model, api_key=key, api_base=endpoint, cache=False)
    dspy.configure(lm=lm, cache=False)
    return lm


def extract(text, examples=None):
    from boundary_reassert import load_boundary_demos
    from context import LLT_CONTEXT
    from dspy_prompts import PromptRouter_ChainOfThought, PromptRouter_LDL_Target

    router = PromptRouter_ChainOfThought()
    demos = load_boundary_demos(examples)
    router.boundary.demos = demos
    decision = dict(router(context=LLT_CONTEXT, text=text))
    target = dict(PromptRouter_LDL_Target()(context=LLT_CONTEXT, text=text))
    return {'decision': decision, 'ldl_target': target}, len(demos)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--note', type=Path, required=True, help='UTF-8 note file')
    parser.add_argument('--output', type=Path, required=True, help='new JSON output file')
    parser.add_argument('--examples', type=Path, help='private ordered boundary demonstration JSON')
    parser.add_argument('--provider', choices=['aihub', 'compatible'], default=None)
    parser.add_argument('--model', help='DSPy model name; default openai/gpt-5')
    parser.add_argument('--base-url', help='OpenAI-compatible endpoint URL')
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name('.env'))
    provider = args.provider or os.environ.get('MODEL_PROVIDER', 'aihub')
    model = args.model or os.environ.get('MODEL_NAME', 'openai/gpt-5')
    base_url = args.base_url or os.environ.get('MODEL_BASE_URL')
    if provider not in ('aihub', 'compatible'):
        parser.error('MODEL_PROVIDER must be aihub or compatible')
    if args.output.exists():
        parser.error('output already exists; choose a new file')
    try:
        text = args.note.read_text(encoding='utf-8')
        if not text.strip():
            parser.error('note is empty')
        from boundary_reassert import load_boundary_demos
        demos = load_boundary_demos(args.examples)
    except (OSError, ValueError, KeyError, TypeError):
        parser.error('cannot read note or demonstrations; check paths and example format')
    if not demos:
        print('Running without the nine private study demonstrations. Published performance is not reproduced by this configuration.', file=sys.stderr)
    try:
        configure_model(provider, model, base_url)
    except ValueError as exc:
        parser.error(str(exc))
    except Exception:
        print('Model configuration failed; request details are suppressed.', file=sys.stderr)
        return 1
    try:
        result, count = extract(text, args.examples)
    except Exception:
        # Provider exceptions can embed submitted notes or credentials.
        print('Inference failed. Check model access and configuration; request details are suppressed.', file=sys.stderr)
        return 1
    result['configuration'] = {
        'version': 'v83', 'model': model, 'provider': provider,
        'demonstration_count': count,
        'demonstrations_sha256': hashlib.sha256(args.examples.read_bytes()).hexdigest() if args.examples else None,
        'dependencies': {name: importlib.metadata.version(name) for name in ('dspy', 'litellm', 'openai', 'httpx', 'python-dotenv')},
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally overwriting an existing record.
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print('Saved prediction.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
