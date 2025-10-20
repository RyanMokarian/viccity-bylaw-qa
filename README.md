# 🧠 Municipal Bylaw Q&A Extraction using DistilBERT

## Overview
This project demonstrates how **machine learning** can automatically find answers to typical regulatory or bylaw questions from publicly available PDF and webpage documents — without the need for manual keyword search or paid APIs.

The system takes a list of predefined questions (for example:  
- *Is a business licence required to operate a short-term rental?*  
- *Is a provincial registration required?*  
- *What is the maximum number of nights an entire home may be rented per year?*)  

and uses a **DistilBERT** question-answering model to locate and extract the exact sentences that contain those answers from a combination of local PDF bylaws and government webpages.

---

## Key Features

- **Automated Q&A Extraction**  
  Uses a pretrained [Hugging Face DistilBERT](https://huggingface.co/distilbert-base-uncased-distilled-squad) model fine-tuned for question answering (SQuAD) to identify the most relevant text span for each question.

- **Dynamic Results**  
  Every time the script runs, it downloads or reads the latest public bylaw PDF and government webpages.  
  If any of those sources change, the answers in the JSON output (`indicators_ml.json`) automatically update.

- **Expandable Question Set**  
  Typical questions are defined in [`questions.py`](./questions.py).  
  New questions can be easily added or existing ones modified — the same model and logic will handle them without retraining.

- **Multi-source Search**  
  Combines paragraphs from both local PDF documents and live webpages.  
  A TF-IDF retriever ranks relevant paragraphs before running the ML model to improve precision.

- **Free and Offline-Capable**  
  Unlike using ChatGPT or other paid LLM APIs, this approach runs **locally** and **for free** once the model is downloaded.

---

## Repository Contents

| File | Description |
|------|--------------|
| **`bylaw_qa_extract.py`** | Main pipeline script: loads sources, runs retrieval, calls the DistilBERT QA model, applies validation rules, and writes structured results to `indicators_ml.json`. |
| **`qa_engine.py`** | Wrapper class around Hugging Face’s DistilBERT Question-Answering pipeline. Handles encoding, batching, and answer span extraction. |
| **`pdf_io.py`** | Utilities for downloading, cleaning, and parsing PDF text; fetching HTML pages; and ranking paragraphs by TF-IDF similarity. |
| **`questions.py`** | Stores typical question templates. Can be edited or extended to support new regulatory or policy domains. |

---

## Example Output

After running:
```bash
python bylaw_qa_extract.py
```

A JSON file is generated:
```json
{
  "requires_license": {
    "value": true,
    "evidence": "A person must not operate or advertise a short-term rental unless a valid business licence has been issued...",
    "source": "https://www.victoria.ca/media/file/short-term-rental-regulation-bylaw-18-036-0"
  },
  "principal_residence_only": {
    "value": true,
    "evidence": "Short-term rentals are only permitted in your principal dwelling unit...",
    "source": "https://www.victoria.ca/building-business/business-licensing/short-term-rentals"
  },
  "max_entire_home_nights": {
    "value": 160,
    "evidence": "While you are away, you can rent your entire unit for no more than 160 nights in a calendar year.",
    "source": "https://www.victoria.ca/building-business/business-licensing/short-term-rentals"
  }
}
```

---

## How It Works

1. **Document Gathering**  
   - Downloads the municipal bylaw PDF (City of Victoria Short-Term Rental Regulation Bylaw).  
   - Fetches related BC Government webpages (host registration, platform requirements, legislation pages).

2. **Text Processing**  
   - Cleans and segments PDF/webpage text into paragraphs and sentences.  
   - Ranks paragraphs using TF-IDF similarity to each question.

3. **Machine Learning QA**  
   - Runs DistilBERT in question-answering mode to predict the exact answer span within each paragraph.

4. **Validation & Output**  
   - Applies regex validators to ensure that evidence sentences logically match each question (e.g., licence, registration, nights).  
   - Outputs structured JSON with answer value, evidence, and source.

---

## Why DistilBERT?

- **Lightweight and Fast**: about 40% smaller and 60% faster than BERT, ideal for local or small-scale automation.  
- **Pretrained QA Capability**: already fine-tuned on SQuAD, enabling accurate span extraction without extra training.  
- **Free and Offline**: runs locally using open-source libraries (`transformers`, `torch`).

While modern tools such as **ChatGPT** or **GPT-4** could also answer these questions, they require API access and cost per token.  
This project provides a **cost-free, reproducible, and privacy-preserving** alternative that can be scheduled or embedded in internal workflows.

---

## Setup

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
*(Typical dependencies: `transformers`, `torch`, `requests`, `pdfminer.six`, `nltk`, `scikit-learn`, `beautifulsoup4`)*

### 3. Run the pipeline
```bash
python bylaw_qa_extract.py
```

### 4. View results
Open the generated `indicators_ml.json` file to see answers, evidence sentences, and source links.

---

## Extending the Project

- Add or modify questions in [`questions.py`](./questions.py).  
- Add new document URLs (PDFs or webpages) to the `EXTRA_URLS` list in `bylaw_qa_extract.py`.  
- Adjust validation patterns in the same script to fit your new domain (for example, zoning, environmental, or licensing bylaws).

---

## License
This repository is provided under the **MIT License**.  
You’re free to use, modify, and adapt it for any municipal or regulatory information-extraction project.

---

## Author
**Ryan Mokarian**  
Data Scientist | Machine Learning | AI Automation  
[GitHub: RyanMokarian](https://github.com/RyanMokarian)
