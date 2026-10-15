"""CORD-v2 extraction: either OCR->text-only LLM or direct vision-language LLM.

Only local Ollama requests; no paid API is involved.
"""
import argparse
import io
import json
import time
from pathlib import Path

from common import Receipt, normalize_ground_truth

VISION_PROMPT = (
    'Read this receipt image. Extract every purchased item name, its printed '
    'line price, and the final transaction total. Do not treat taxes, subtotals, '
    'discount rows or zero-price modifiers as purchased items. Preserve printed '
    'item names and prices. Only extract what is visible. Return valid JSON.'
)
TEXT_PROMPT = (
    'Extract every purchased item name, its printed line price and the final '
    'transaction total from the OCR text below. The OCR may have errors; do not '
    'invent or guess values. Exclude taxes, subtotals, discount rows and zero-price '
    'modifiers from purchased items. Return valid JSON.\n\nOCR TEXT:\n'
)


def image_bytes(image):
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='PNG')
    return buffer.getvalue()


def extract_receipt(image, model: str = 'qwen2.5vl:7b', pipeline: str = 'vlm',
                    ocr_lang: str = 'eng') -> tuple[Receipt, str | None]:
    """Return (validated receipt, intermediate OCR text if used)."""
    import ollama
    if pipeline == 'vlm':
        messages = [{'role': 'user', 'content': VISION_PROMPT,
                     'images': [image_bytes(image)]}]
        ocr_text = None
    elif pipeline == 'ocr_llm':
        import pytesseract
        try:
            ocr_text = pytesseract.image_to_string(image.convert('RGB'), lang=ocr_lang)
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError('Tesseract executable is missing. Install Tesseract OCR and add it to PATH.') from exc
        if not ocr_text.strip():
            raise ValueError('OCR produced no text for this receipt.')
        messages = [{'role': 'user', 'content': TEXT_PROMPT + ocr_text}]
    else:
        raise ValueError(f'Unknown pipeline: {pipeline}')
    response = ollama.chat(model=model, messages=messages,
                           format=Receipt.model_json_schema(), options={'temperature': 0})
    return Receipt.model_validate_json(response.message.content), ocr_text


def run(limit: int, model: str, pipeline: str, output_dir: Path,
        save_images: bool, ocr_lang: str) -> None:
    from datasets import load_dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset('naver-clova-ix/cord-v2', split='validation')
    destination = output_dir / 'predictions.jsonl'
    with destination.open('w', encoding='utf-8') as stream:
        for receipt_id in range(min(limit, len(dataset))):
            print(f'[{pipeline}] receipt {receipt_id + 1}/{limit}', flush=True)
            start = time.perf_counter()
            record = {'receipt_id': receipt_id, 'model': model,
                      'pipeline': pipeline, 'status': 'error'}
            try:
                sample = dataset[receipt_id]
                record['ground_truth'] = normalize_ground_truth(
                    json.loads(sample['ground_truth'])['gt_parse'])
                if save_images:
                    (output_dir / 'images').mkdir(exist_ok=True)
                    sample['image'].save(output_dir / 'images' / f'receipt_{receipt_id:03d}.png')
                prediction, ocr_text = extract_receipt(
                    sample['image'], model, pipeline, ocr_lang)
                record['prediction'] = prediction.model_dump()
                if ocr_text is not None:
                    record['ocr_text'] = ocr_text
                record['status'] = 'ok'
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
                print(record['error'])
            finally:
                record['time_seconds'] = round(time.perf_counter() - start, 2)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
    print(f'Wrote {destination}')
    print(f'Next: python evaluate.py --input {destination} --output-dir {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run one of two CORD-v2 local inference pipelines')
    parser.add_argument('--pipeline', choices=['vlm', 'ocr_llm'], default='vlm')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--model', default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--save-images', action='store_true')
    parser.add_argument('--ocr-lang', default='eng')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    model = args.model or ('qwen2.5vl:7b' if args.pipeline == 'vlm' else 'qwen2.5:7b')
    output = args.output_dir or Path('results') / args.pipeline
    run(args.limit, model, args.pipeline, output, args.save_images, args.ocr_lang)
