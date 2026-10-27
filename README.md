# HelloWebsite

This repo holds two separate projects.

## 1. Business landing page (HTML/CSS)

A responsive, multi-section landing page for a digital agency, built with plain HTML and CSS: navigation bar, hero section, service cards, portfolio gallery, client logos and a contact footer.

- `index.html` – page markup
- `style.css` – all styling
- `images/` – images and icons

**Live demo:** https://malaikah540.github.io/HelloWebsite/

To run it locally, just open `index.html` in a browser.

## 2. FAIRPAY AI – anomaly detection proof of concept (Python)

Located in [`fairpay_ai/`](fairpay_ai/). A machine-learning pipeline that looks for "loyalty penalty" pricing (long-standing customers paying unusually high renewal prices) in UK broadband, mobile and subscription markets.

Pipeline steps:

1. **Dataset generation** – 10,000 synthetic pricing records with injected anomalies
2. **Model training** – Isolation Forest (scikit-learn)
3. **Evaluation** – metrics, confusion matrix and contamination sensitivity plot
4. **Fairness audit** – false-positive rate compared across UK regions
5. **Report** – consumer alert CSV

Sample outputs (charts and CSV) are in [`fairpay_ai/outputs/`](fairpay_ai/outputs/).

### Run it

```bash
cd fairpay_ai
pip install -r requirements.txt
python main.py
```

**Tech:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn
