# Document Intelligence: OCR + LLM vs Vision-Language Model

[![Live interactive dashboard](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://cord-receipt.streamlit.app/)

**Live demo:** https://cord-receipt.streamlit.app/

A lightweight document-information-extraction application comparing two approaches to understanding receipts. The project uses **local Ollama models**, a shared JSON output schema, automated ground-truth evaluation, and a hosted **interactive replay dashboard**. No paid inference API is required.

> **Public demo vs local inference:** The Streamlit Cloud dashboard lets visitors explore saved predictions, review ground truth, inspect OCR output, and compare benchmark runs. It **does not execute inference** or host the local Ollama models. Live inference is available when the project is run locally with Ollama.

## Pipelines

### A — OCR + text-only LLM

```text
CORD-v2 receipt image
          |
          v
   Tesseract OCR
          |  recognised plain text
          v
 Qwen2.5 7B (Ollama)
          |
          v
Pydantic validated JSON
          |
          v
Ground-truth evaluation
```

Tesseract reads the image and passes the recognised text to `qwen2.5:7b`. The model identifies purchased items, their prices, and the final total. The OCR output is saved so recognition errors can be investigated separately from LLM extraction errors.

### B — Vision-language model (VLM)

```text
CORD-v2 receipt image
          |
          v
 Qwen2.5-VL 7B (Ollama)
          |
          v
Pydantic validated JSON
          |
          v
Ground-truth evaluation
```

`qwen2.5vl:7b` receives the image directly, without a separate Tesseract stage. Both pipelines produce the same simplified JSON schema:

```json
{
  "items": [{"name": "Example product", "price": "23000"}],
  "total": "23000"
}
```

The source is the [CORD-v2 dataset](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (receipt images with annotated fields). The goal is to demonstrate extraction and **comparative evaluation**, not to replicate the official CORD benchmark.

## Results

### Overall evaluated runs

The first table uses all **successful evaluations within each pipeline**, so sample sizes differ. These are exploratory validation-set results from the current saved experiment.

| Metric | VLM | OCR + LLM |
|---|---:|---:|
| Attempted receipts | 50 | 50 |
| Successful evaluations | 50 | 39 |
| Failed evaluations | 0 | 11 |
| Item precision | 78.23% | 15.58% |
| Item recall | 81.51% | 13.04% |
| Item F1 | 79.84% | 14.20% |
| Total-price accuracy | 100% | 21.62% |
| Mean time / successful receipt | 5.95 s | 3.97 s |

### Fair comparison: same 39 successful receipt IDs

For a direct paired comparison, the dashboard **intersects successfully evaluated receipt IDs** and recalculates *micro-aggregated* item precision/recall/F1 and total accuracy. Failed runs are shown separately, not silently included as correct.

| Metric | VLM | OCR + LLM |
|---|---:|---:|
| Shared successful receipt IDs | 39 | 39 |
| True positives | 75 | 12 |
| False positives | 19 | 65 |
| False negatives | 17 | 80 |
| Item precision | **79.79%** | 15.58% |
| Item recall | **81.52%** | 13.04% |
| Item F1 | **80.65%** | 14.20% |
| Total-price accuracy | **100%** | 21.62% |
| Mean time / successful receipt | 5.61 s | 3.97 s |

**Interpretation:** On these CORD-v2 validation receipts and with these specific configurations, the VLM obtained substantially higher extraction scores. This is **not** evidence that all VLMs outperform all OCR-based approaches. The 11 unsuccessful OCR runs are excluded from paired extraction metrics; their completion failures remain important reliability information. Small sample size, schema simplification and exact matching also limit generalisation.

### Failure analysis

Examples from the OCR pipeline:

- Receipt 0: Tesseract found prices but dropped the product names.
- Receipt 1: OCR read price/payment/change amounts but missed the item name; the LLM treated payment as an item and total.
- Receipt 3: OCR captured an item and price, but the text LLM generated an invalid mathematical expression instead of copying a price.
- Receipts 4 and 7: OCR missed important item or price text.
- Eleven receipts generated empty OCR output in the first 50-receipt run.

These observations help distinguish **text recognition**, **layout preservation**, and **LLM extraction** failures. The benchmark uses simplified exact matching after price and text normalisation; price semantics (unit vs line totals) can still introduce mismatches.

## Dashboard

The app has two tabs:

- **Receipt explorer:** select a pipeline and receipt, view source image, predicted items and totals, labelled ground truth, errors and optional OCR text.
- **Pipeline comparison:** view (1) all-successful-run metrics with completion counts and (2) a **fair paired comparison calculated from shared receipt IDs**. Results and comparisons can be downloaded as CSV.

The dashboard reads `results/` in local mode and `demo_data/` in public replay mode. To publish new results, copy the evaluated `results/<pipeline>/` folders—including saved receipt images—to `demo_data/<pipeline>/` and commit the demo assets.

## Run locally

Install Python dependencies and the Tesseract executable. On Ubuntu:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
python -m pip install -r requirements-local.txt
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
```

Run both pipelines against the same first 50 validation receipts:

```bash
python inference.py --pipeline ocr_llm --limit 50 --save-images
python evaluate.py --input results/ocr_llm/predictions.jsonl --output-dir results/ocr_llm

python inference.py --pipeline vlm --limit 50 --save-images
python evaluate.py --input results/vlm/predictions.jsonl --output-dir results/vlm
```

Start the interactive app:

```bash
streamlit run app.py
```

Test the cloud-compatible saved-results interface locally:

```bash
DEMO_MODE=true streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Commit `app.py`, `inference.py`, `evaluate.py`, `common.py`, `requirements.txt`, and the permitted `demo_data/` assets to a GitHub repository.
2. Deploy `app.py` on [Streamlit Community Cloud](https://share.streamlit.io/).
3. Set `DEMO_MODE = "true"` in the app's Streamlit Cloud secrets.
4. Confirm the public app shows both pipelines and their receipt images; update the **Live demo** link at the top of this README to your actual URL.

**Important:** `results/` usually stays untracked. `demo_data/` must contain the files the hosted app reads. Check the CORD-v2 dataset's licence and attribution before redistributing images.

## Repository layout

```text
app.py                    Streamlit user interface
inference.py              Local OCR+LLM and VLM inference
common.py                 Shared Pydantic schema and GT adapter
evaluate.py               Offline metrics calculation
requirements.txt          Lightweight hosted-dashboard dependencies
requirements-local.txt    Local inference dependencies
demo_data/                Saved public-demo predictions and images
results/                  Locally generated experimental outputs (usually ignored)
```

## Limitations and next steps

This portfolio prototype evaluates just receipt items and totals rather than every field in CORD-v2. The baseline OCR system uses plain-text Tesseract output without layout coordinates, and the LLM can return semantically incorrect strings despite schema validation. Latency depends on hardware, model warm-up and caching. Further work could include better OCR preprocessing, positional OCR, document-type expansion, and larger held-out benchmarks—but these are intentionally outside this small MVP.
