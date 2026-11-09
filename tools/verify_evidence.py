"""Validate saved evidence using the standard library; never load model weights."""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]

def main():
    for path in ROOT.rglob('*.py'):
        if '.git' not in path.parts:
            ast.parse(path.read_text(), filename=str(path))
    tree = ast.parse((ROOT / 'experiments/benchmark_sft_grpo.py').read_text())
    # Compile only the reviewed pure scoring definitions; never execute imports/model code.
    scope = {'re': re}
    questions = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {'repetition_flag', 'exact_match'}:
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<scoring>', 'exec'), scope)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'QUESTIONS' for t in node.targets):
            questions = ast.literal_eval(node.value)
    refusal = re.compile(r"\b(?:i cannot|i can't|i am unable|i'm unable|i cannot provide|i can't answer|i'm sorry, but i cannot)\b", re.I)
    rows = [json.loads(line) for line in (ROOT / 'experiments/results/benchmark_sft_grpo.jsonl').read_text().splitlines()]
    assert len(rows) == len({(r['model'], r['id']) for r in rows}) == 90
    summary = {}
    for model in ['SFT', 'DPO', 'GRPO']:
        subset = [r for r in rows if r['model'] == model]
        assert {r['id'] for r in subset} == set(range(1, 31))
        for r in subset:
            assert (r['category'], r['question'], r['expected']) == questions[r['id'] - 1]
            assert r['exact_match'] == scope['exact_match'](r['answer'], r['expected'])
            assert r['possible_repetition'] == scope['repetition_flag'](r['answer'])
            assert r['possible_refusal'] == bool(refusal.search(r['answer']))
        summary[model] = {'completed': len(subset), 'strict_exact_matches': sum(r['exact_match'] or False for r in subset), 'strict_exact_match_total': sum(r['exact_match'] is not None for r in subset), 'possible_refusals': sum(r['possible_refusal'] for r in subset), 'possible_repetitions': sum(r['possible_repetition'] for r in subset)}
    assert summary == json.loads((ROOT / 'experiments/results/benchmark_summary.json').read_text())
    log = (ROOT / 'evidence/training/grpo_full_resume.log').read_bytes().split(b'\n')
    assert b'Epoch:[1/1](19502/19502)' in log[973]
    assert b'(317560/317560)' in (ROOT / 'evidence/training/pretrain_full.log').read_bytes()
    assert b'(56295/56295)' in (ROOT / 'evidence/training/sft_full.log').read_bytes()
    manifest = ROOT / 'docs/PREPARED_MANIFEST.json'
    if manifest.exists():
        for item in json.loads(manifest.read_text()):
            assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    print('PASS: Python syntax, 90 prompts/scoring flags/summary, training endpoints and prepared file hashes.')

if __name__ == '__main__':
    main()
