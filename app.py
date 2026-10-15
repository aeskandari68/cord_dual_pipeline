"""Streamlit dual pipeline demo. Public mode loads saved results, never calls Ollama."""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluate import evaluate_one, load_records

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / 'results'
def public_demo_mode():
    setting = os.getenv('DEMO_MODE')
    if setting is None:
        try:
            setting = str(st.secrets.get('DEMO_MODE', 'false'))
        except Exception:
            setting = 'false'
    return setting.lower() in {'true', '1', 'yes'}


DEMO_MODE = public_demo_mode()
PIPELINES = {'vlm': 'Vision-language model', 'ocr_llm': 'OCR + text-only LLM'}

st.set_page_config(page_title='CORD: OCR vs VLM', page_icon='🧾', layout='wide')
st.title('🧾 Document Intelligence: OCR vs VLM')
st.caption('Two local inference pipelines · shared JSON schema · same CORD-v2 ground truth')
st.info('**Portfolio replay mode:** viewing recorded predictions, not running live inference.' if DEMO_MODE
        else '**Local mode:** browse recorded predictions or run fresh Ollama inference.')


@st.cache_data(show_spinner=False)
def records_for(path: str):
    return load_records(Path(path)) if Path(path).exists() else []


@st.cache_data(show_spinner=False)
def metrics_for(directory: str):
    directory = Path(directory)
    summary = directory / 'summary.json'
    return json.loads(summary.read_text(encoding='utf8')) if summary.exists() else None


@st.cache_resource(show_spinner=False)
def get_dataset():
    from datasets import load_dataset
    return load_dataset('naver-clova-ix/cord-v2', split='validation')


def locate_images(receipt_id, folders):
    for folder in folders:
        for path in (folder / 'images' / f'receipt_{receipt_id:03d}.png',
                     folder / f'receipt_{receipt_id:03d}.png'):
            if path.exists():
                return path
    return None


def table_items(value):
    return pd.DataFrame([{'Item': row.get('name', ''), 'Price': row.get('price', '')}
                         for row in value.get('items', [])], columns=['Item', 'Price'])


def folders(pipeline):
    root = DEMO if DEMO_MODE else ROOT / 'results'
    primary = root / pipeline
    # Older releases stored the single VLM experiment in the parent directory.
    if pipeline == 'vlm' and not (primary / 'predictions.jsonl').exists():
        return root
    return primary


def show_prediction(pred, gt):
    a, b = st.columns(2)
    with a:
        st.markdown('#### Prediction')
        st.dataframe(table_items(pred), hide_index=True, use_container_width=True)
        st.write('**Total:**', pred.get('total'))
    with b:
        st.markdown('#### Ground truth')
        st.dataframe(table_items(gt), hide_index=True, use_container_width=True)
        st.write('**Total:**', gt.get('total'))
    outcome = evaluate_one(pred, gt)
    cols = st.columns(4)
    cols[0].metric('Item precision', f"{outcome['precision']:.1%}")
    cols[1].metric('Item recall', f"{outcome['recall']:.1%}")
    cols[2].metric('Item F1', f"{outcome['f1']:.1%}")
    cols[3].metric('Total correct', 'Yes' if outcome['total_correct'] is True else 'No'
                   if outcome['total_correct'] is False else 'No GT')
    left, right = st.columns(2)
    with left:
        st.write('**Extra predictions**')
        st.json(outcome['extra_items'])
    with right:
        st.write('**Missed GT items**')
        st.json(outcome['missing_items'])
    st.download_button('Download prediction JSON', json.dumps(pred, indent=2, ensure_ascii=False),
                       file_name='prediction.json', mime='application/json')


with st.sidebar:
    st.header('Pipelines')
    pipeline = st.radio('Select approach', options=list(PIPELINES),
                        format_func=lambda key: PIPELINES[key])
    st.write('**VLM:** receipt image → Qwen2.5-VL → JSON')
    st.write('**OCR + LLM:** image → Tesseract → Qwen2.5 (text) → JSON')
    st.caption('Simplified exact matching; not the official CORD benchmark metric.')
    if not DEMO_MODE:
        model = st.text_input('Local Ollama model',
                              value='qwen2.5vl:7b' if pipeline == 'vlm' else 'qwen2.5:7b',
                              key=f'model_{pipeline}')
    st.markdown('[CORD-v2 dataset](https://huggingface.co/datasets/naver-clova-ix/cord-v2)')

explore, compare = st.tabs(['🔎 Receipt explorer', '📊 Pipeline comparison'])

with explore:
    directory = folders(pipeline)
    records = [r for r in records_for(str(directory / 'predictions.jsonl'))
               if r.get('status') == 'ok' and isinstance(r.get('prediction'), dict)]
    ids = sorted({int(r['receipt_id']) for r in records})
    if not records:
        st.warning(f'No saved {PIPELINES[pipeline]} predictions yet. Run inference and evaluation locally, then copy the result files to the matching folder.')
    if DEMO_MODE:
        receipt_id = st.selectbox('Receipt', options=ids) if ids else None
    else:
        receipt_id = st.number_input('CORD validation receipt ID', min_value=0, max_value=99, value=ids[0] if ids else 0)
    if receipt_id is not None:
        record = next((r for r in records if int(r['receipt_id']) == receipt_id), None)
        pic_col, detail_col = st.columns([1, 1])
        image_path = locate_images(receipt_id, [directory, DEMO / 'vlm', DEMO, ROOT / 'results' / 'vlm', ROOT / 'results'])
        with pic_col:
            if image_path:
                st.image(str(image_path), caption=f'CORD validation receipt {receipt_id}', use_container_width=True)
            elif DEMO_MODE:
                st.info('Receipt PNG not included. Add PNGs from a local `--save-images` run to demo_data/<pipeline>/images/.')
            else:
                try:
                    st.image(get_dataset()[receipt_id]['image'], use_container_width=True)
                except Exception as exc:
                    st.warning(f'Image unavailable: {exc}')
        with detail_col:
            if record:
                st.write('**Recorded model:**', record.get('model'))
                st.write('**Execution time:**', record.get('time_seconds'), 'seconds')
                st.caption('Saved prediction — no inference request made.')
            if not DEMO_MODE and st.button('Run live extraction', type='primary'):
                try:
                    from inference import extract_receipt
                    from common import normalize_ground_truth
                    sample = get_dataset()[receipt_id]
                    with st.spinner('Running local inference...'):
                        import time
                        start = time.perf_counter()
                        prediction, ocr_text = extract_receipt(sample['image'], model, pipeline)
                        elapsed = time.perf_counter() - start
                    gt = normalize_ground_truth(json.loads(sample['ground_truth'])['gt_parse'])
                    st.session_state[f'live_{pipeline}'] = {
                        'receipt_id': receipt_id, 'prediction': prediction.model_dump(),
                        'ground_truth': gt, 'ocr_text': ocr_text, 'time_seconds': elapsed,
                    }
                except Exception as exc:
                    st.error(f'Inference error: {exc}')
        latest = st.session_state.get(f'live_{pipeline}') if not DEMO_MODE else None
        selected = latest if latest and latest['receipt_id'] == receipt_id else record
        if selected:
            st.divider()
            show_prediction(selected['prediction'], selected['ground_truth'])
            if selected.get('ocr_text') is not None:
                with st.expander('Inspect intermediate OCR text'):
                    st.code(selected['ocr_text'], language='text')

with compare:
    snapshots = []
    for key, label in PIPELINES.items():
        directory = folders(key)
        summary = metrics_for(str(directory))
        if summary:
            snapshots.append({
                'Approach': label, 'Model': summary.get('model'),
                'Receipts': summary.get('succeeded'),
                'Item precision': summary.get('item_micro_precision'),
                'Item recall': summary.get('item_micro_recall'),
                'Item F1': summary.get('item_micro_f1'),
                'Total accuracy': summary.get('total_accuracy'),
                'Mean seconds': summary.get('mean_seconds_per_success'),
            })
    if not snapshots:
        st.warning('No evaluated results found. Run evaluate.py for each approach first.')
    else:
        df = pd.DataFrame(snapshots)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.bar_chart(df.set_index('Approach')[['Item precision', 'Item recall', 'Item F1']])
        st.download_button('Download comparison CSV', df.to_csv(index=False), 'comparison.csv', 'text/csv')
        if len(snapshots) < 2:
            st.info('Only the VLM baseline is available. Run OCR + LLM locally to enable a real two-pipeline comparison.')
        else:
            st.caption('Compare on identical validation receipt IDs for fairness. Performance timings depend on hardware and runtime settings.')
    st.markdown('**Experiment design:** identical receipt images, shared JSON schema and matching metrics. '
                'The OCR pipeline has a distinct recognition stage whose mistakes propagate to the text-only LLM.')
