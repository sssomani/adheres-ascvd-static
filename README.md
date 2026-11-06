# ADHERES ASCVD

ADHERES ASCVD uses language models to extract lipid-management decisions from clinical notes. It helps researchers study documented treatment goals, changes in lipid-lowering therapy (LLT), and reasons for not intensifying treatment in patients with atherosclerotic cardiovascular disease (ASCVD).

This repository provides the extraction prompts and Python code to analyze a note and return four structured outcomes:

- Whether an LDL-C target was discussed.
- The LDL-C target value.
- Whether LLT was changed.
- Reasons for LLT non-escalation.

Only code and synthetic examples are included. Clinical data are not distributed.

## Quick start

Tested with Python 3.11. The default model is GPT-5 through Stanford AI Hub, which requires authorized access and network connectivity.

```bash
git clone https://github.com/sssomani/adheres-ascvd-static.git
cd adheres-ascvd-static
python3 -m venv .venv
source .venv/bin/activate
pip install --require-hashes -r requirements.lock
cp .env.example .env
```

Set `SECURE_GPT_KEY` in your local `.env`, then run the synthetic example:

```bash
python src/run.py --note examples/note.txt --output outputs/prediction.json
```

To analyze your own note, replace `--note` with the path to a UTF-8 text file. The script writes JSON and refuses to overwrite an existing output.

### Use another model

Set `OPENAI_API_KEY` in your local `.env` and supply an OpenAI-compatible endpoint:

```bash
python src/run.py --provider compatible --model openai/YOUR_MODEL \
  --base-url https://YOUR_HOST/v1 \
  --note examples/note.txt --output outputs/alternative.json
```

## Output

| Outcome | JSON field |
| --- | --- |
| LDL-C target discussed | `ldl_target.ldl_target_discussed` |
| Target value | `ldl_target.ldl_target_value` |
| LLT change | `decision.llt_change` |
| Reasons | `decision.llt_reasons` |

Target values are `<55`, `<70`, `<100`, `None` (unspecified), or `Other`; `other_target` supplies additional detail. Interpret the value together with whether a target was discussed. Reasons describe escalation when therapy is intensified and non-escalation otherwise. Unusable notes receive no reasons.

The output also includes supporting model fields and configuration metadata. Prompt definitions and allowed labels are in [src/dspy_prompts.py](src/dspy_prompts.py).

## Reproducibility and data

Dependencies are pinned in `requirements.lock`. The output records software versions, source hashes, and demonstration settings.

The nine clinical examples used as few-shot demonstrations in the study are not included. The default run uses no demonstrations and does **not** reproduce the study's full configuration or establish its reported performance. Authorized users can supply private examples with `--examples /secure/boundary_demos.json`: an ordered JSON array containing `text`, `corrected_boundary_labels`, and `boundary_evidence`. Model changes and hosted-model updates can also affect results.

Keep real notes, private examples, credentials, and generated outputs out of Git. Notes and examples are sent to the selected model endpoint; use an environment approved for your data. This is research software.

## License

[MIT](LICENSE).
