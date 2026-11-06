# ADHERES ASCVD — static note extraction

The final **v83** prompts and inference logic for extracting LDL-C target discussion, LDL-C target value, lipid-lowering therapy (LLT) change, and reasons for non-escalation from a clinical note. Uses GPT-5 through Stanford AI Hub by default; supports an alternative OpenAI-compatible model endpoint.

## Run

Tested with Python 3.11:

```bash
git clone https://github.com/sssomani/adheres-ascvd-static.git
cd adheres-ascvd-static
python3 -m venv .venv
source .venv/bin/activate
pip install --require-hashes -r requirements.lock
cp .env.example .env
# Set SECURE_GPT_KEY in .env; AI Hub requires authorized access/network connectivity.
python run.py --note examples/note.txt --output outputs/prediction.json
```

For another endpoint, set `OPENAI_API_KEY` and run:

```bash
python run.py --provider compatible --model openai/YOUR_MODEL \
  --base-url https://YOUR_HOST/v1 \
  --note examples/note.txt --output outputs/alternative.json
```

The bundled note is entirely synthetic. Inputs are plain UTF-8 text; outputs are JSON. Existing outputs are never overwritten. Model calls use the configured provider and may incur charges.

## Outputs

| Outcome | JSON field | Values |
| --- | --- | --- |
| LDL-C target discussed | `ldl_target.ldl_target_discussed` | `discussed`, `not discussed` |
| Target value | `ldl_target.ldl_target_value` | `<55`, `<70`, `<100`, `None`, `Other`; see `other_target` for Other |
| LLT change | `decision.llt_change` | `escalation`, `de-escalation`, `change`, `no change`, `bad note` |
| Reasons | `decision.llt_reasons` | List of v83 reason labels, defined in `dspy_prompts.py` |

Reasons describe escalation on that route and non-escalation for `de-escalation`, `change`, or `no change`. `bad note` receives no reasons. The `None` target label is a string for an unspecified target; interpret it together with the discussion field. Original auxiliary fields are retained, including `other_reasons`, reasoning, and non-escalation LDL assessment fields. Internal evidence is filtered and removed, as in v83. Configuration records software versions, source hashes, and the demonstration count/hash without copying the input note.

## Reproducing v83

The seven source modules retain v83's prompts, routing, and gate order. `context.py` contains the exact LLT context from its inference notebook. The non-escalation route performs boundary reclassification, LDL goal filtering, evidence filtering, missed-lab-deferral detection, lab-deferral verification, and “other” suppression in that order. Verification failures preserve the preceding prediction, matching the original implementation. `source_manifest.json` records source hashes and packaging changes.

**The nine original few-shot demonstrations contain clinical training-note excerpts and are not distributed.** Without them, this runs the v83 code with an empty demonstration set; it does **not** reproduce the study's full configuration or establish its published performance. Authorized users can supply their private ordered examples:

```bash
python run.py --note /secure/note.txt --examples /secure/boundary_demos.json \
  --output /secure/prediction.json
```

Examples must be a JSON array of objects with `text`, `corrected_boundary_labels` (a list), and `boundary_evidence` (a label-to-quote-list mapping). Their order is preserved. No examples are loaded implicitly. Alternative models or demonstrations change the configuration. `requirements.lock` pins direct and transitive dependencies with package hashes (resolved for Python 3.11); `requirements.txt` lists the direct dependencies. These pins describe the tested release environment, not a recovered historical environment; hosted model outputs are not guaranteed to be identical across runs. No temperature or token limit overrides are added to the notebook's model configuration.

## Data and checks

No clinical datasets, original demonstrations, predictions, credentials, notebooks, or SQL are included. Keep real notes and outputs outside this repository: model responses can reproduce sensitive text. The runner disables DSPy caches and routine SDK logs and suppresses inference exception details; it sends notes and supplied demonstrations to the selected endpoint. Use an endpoint authorized for your data. This is research software.

Run offline checks with `python -m unittest discover -s tests -v`. The smoke tests exercise the real CLI and both provider transports against a localhost mock API using synthetic text; they require permission to bind a local port and never contact a live model.

License: [MIT](LICENSE). Manuscript citation/link will be added when supplied.
