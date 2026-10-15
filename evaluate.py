"""Evaluate saved predictions offline, without importing or calling Ollama."""
import argparse
import json
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
from common import Receipt


def normalize_text(value) -> str:
    return ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().split())


def normalize_price(value) -> Decimal | None:
    """Heuristic for CORD's Indonesian receipts, not a universal locale parser."""
    if value is None:
        return None
    text = re.sub(r'[^\d.,-]', '', str(value).strip())
    if not text:
        return None
    # e.g., 20.000 or 20,000; 1.234.567 or 1,234,567
    if re.fullmatch(r'-?\d{1,3}([.,]\d{3})+', text):
        text = text.replace('.', '').replace(',', '')
    elif ',' in text and '.' not in text:
        # CORD's comma-only prices are normally thousand-separated.
        text = text.replace(',', '')
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def count_items(items) -> Counter:
    return Counter((normalize_text(item['name']), normalize_price(item.get('price'))) for item in items)


def describe(counter: Counter) -> list[dict]:
    return [
        {'name': name, 'price': str(price) if price is not None else None}
        for (name, price) in counter.elements()
    ]


def evaluate_one(prediction: dict, gt: dict) -> dict:
    pred = Receipt.model_validate(prediction)
    pred_counts = count_items([i.model_dump() for i in pred.items])
    gt_counts = count_items(gt.get('items', []))
    tp = sum((pred_counts & gt_counts).values())
    fp_counter = pred_counts - gt_counts
    fn_counter = gt_counts - pred_counts
    fp = sum(fp_counter.values())
    fn = sum(fn_counter.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    expected = normalize_price(gt.get('total'))
    predicted = normalize_price(pred.total)
    return {
        'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4),
        'tp': tp, 'fp': fp, 'fn': fn,
        'total_correct': predicted == expected if expected is not None else None,
        'predicted_total': str(predicted) if predicted is not None else None,
        'expected_total': str(expected) if expected is not None else None,
        'extra_items': describe(fp_counter), 'missing_items': describe(fn_counter),
    }


def load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == '.csv':
        # Migration support: evaluate prior base.py results.csv without rerunning Ollama.
        rows = pd.read_csv(path).to_dict(orient='records')
        output = []
        for row in rows:
            record = {'receipt_id': row['receipt_id'], 'model': row.get('model'),
                      'time_seconds': row.get('time_seconds'), 'status': row.get('status', 'ok')}
            try:
                record['prediction'] = json.loads(row['prediction'])
                record['ground_truth'] = json.loads(row['ground_truth'])
            except (ValueError, TypeError, KeyError):
                record['status'] = 'error'
                record['error'] = str(row.get('error', 'missing prediction or GT'))
            output.append(record)
        return output
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def run(input_path: Path, output_dir: Path) -> None:
    records = load_records(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        row = {key: record.get(key) for key in ('receipt_id', 'model', 'status', 'time_seconds')}
        row['error'] = record.get('error', '')
        if record.get('status') == 'ok':
            try:
                metrics = evaluate_one(record['prediction'], record['ground_truth'])
                row.update(metrics)
                row['prediction'] = json.dumps(record['prediction'], ensure_ascii=False)
                row['ground_truth'] = json.dumps(record['ground_truth'], ensure_ascii=False)
                row['extra_items'] = json.dumps(metrics['extra_items'], ensure_ascii=False)
                row['missing_items'] = json.dumps(metrics['missing_items'], ensure_ascii=False)
            except Exception as exc:
                row['status'] = 'error'
                row['error'] = f'Evaluation error: {type(exc).__name__}: {exc}'
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'results.csv', index=False)
    success = df[df['status'] == 'ok'] if not df.empty else df
    tp = int(success['tp'].sum()) if not success.empty else 0
    fp = int(success['fp'].sum()) if not success.empty else 0
    fn = int(success['fn'].sum()) if not success.empty else 0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    totals = success['total_correct'].dropna() if not success.empty else pd.Series(dtype=bool)
    summary = {
        'model': success['model'].iloc[0] if not success.empty else None,
        'pipeline': records[0].get('pipeline', 'vlm') if records else None,
        'attempted': len(df), 'succeeded': len(success), 'failed': len(df) - len(success),
        'item_micro_precision': round(precision, 4), 'item_micro_recall': round(recall, 4),
        'item_micro_f1': round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
        'total_accuracy': round(float(totals.astype(bool).mean()), 4) if len(totals) else None,
        'total_evaluated': len(totals),
        'mean_seconds_per_success': round(float(pd.to_numeric(success['time_seconds']).mean()), 2) if not success.empty else None,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(df.reindex(columns=['receipt_id', 'status', 'precision', 'recall', 'f1', 'total_correct']).to_string(index=False))
    print('\nSummary:\n' + json.dumps(summary, indent=2))
    print(f'\nSaved {output_dir / "results.csv"} and {output_dir / "summary.json"}')
    print('Note: simplified name+price exact match; not the official CORD benchmark metric.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate cached CORD-v2 predictions, offline')
    parser.add_argument('--input', type=Path, default=Path('results/predictions.jsonl'))
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'Input file not found: {args.input}')
    run(args.input, args.output_dir)
