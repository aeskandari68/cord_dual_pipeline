# CORD-v2: OCR + Text LLM vs Vision-Language Model

Small local document-information-extraction portfolio experiment, with a public replay dashboard. **No paid APIs.**

## Two approaches

1. **OCR + text-only LLM:** receipt image → Tesseract OCR → Qwen2.5 7B via local Ollama → Pydantic JSON.
2. **Vision-language model:** receipt image → Qwen2.5-VL 7B via local Ollama → same Pydantic JSON.

Both approaches extract `{items: [{name, price}], total}` from the **same CORD-v2 validation split**. `evaluate.py` computes simplified exact normalized item-name/price precision, recall and F1 plus total-price accuracy and runtime. This is **not official CORD scoring**.

## Set up local inference

```bash
pip install -r requirements-local.txt
ollama pull qwen2.5vl:7b
ollama pull qwen2.5:7b
```

Also install the **Tesseract executable** (not just `pytesseract`) and make sure `tesseract --version` runs in your shell.

- Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`
- Windows: install a trusted Tesseract Windows distribution and add its folder to PATH.

## Run both on the same 10 validation receipts

```bash
python inference.py --pipeline vlm --limit 10 --save-images
python evaluate.py --input results/vlm/predictions.jsonl --output-dir results/vlm

python inference.py --pipeline ocr_llm --limit 10 --save-images
python evaluate.py --input results/ocr_llm/predictions.jsonl --output-dir results/ocr_llm
```

The outputs are stored separately, so one model never overwrites another.

Run Streamlit locally:

```bash
streamlit run app.py
```

The **Receipt explorer** lets you inspect predictions and GT and run either model live. For the OCR pipeline, inspect its intermediate OCR text in an expander. The **Pipeline comparison** tab shows aggregate scores, with no new inference calls.

## Free hosted portfolio version

Streamlit Community Cloud does not host Ollama/GPU. Deploy **saved inference results** instead. Set `DEMO_MODE = "true"` in Streamlit Community Cloud **Advanced settings → Secrets** or set the `DEMO_MODE=true` environment variable. The app reads both. Public mode never invokes Ollama or downloads the Hugging Face dataset.

Install only `requirements.txt` in hosted mode. Copy the completed outputs:

```
demo_data/
    vlm/
        predictions.jsonl
        results.csv
        summary.json
        images/receipt_000.png ...
    ocr_llm/
        predictions.jsonl
        results.csv
        summary.json
        images/receipt_000.png ...
```

For compatibility the app also reads previous VLM-only demo data stored directly inside `demo_data/`. That existing baseline is included in this project, but **no OCR predictions are fabricated**; the comparison activates once you run the second pipeline.

Images must be copied from your local `results/<pipeline>/images/` directory. Review the dataset license and attribution before sharing images publicly.

## Notes on benchmarking

- Use the **same receipt IDs** (e.g., `--limit 10`) for both approaches.
- `time_seconds` is end-to-end per sample, including OCR for the OCR pipeline; this is useful but can be influenced by caching and warm-up.
- Item prices may represent unit or line price depending on receipt format. Manual error analysis remains useful.
- The existing VLM baseline is a saved run; results for OCR + LLM must be generated on your workstation.
- The 10-receipt demo is exploratory, not a scientifically robust comparison. Expand to 50 or 100 validation receipts later.
