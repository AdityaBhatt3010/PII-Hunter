# 🕵️‍♂️ PII Hunter

> **Hunt. Detect. Tag.**
> An advanced Personally Identifiable Information (PII) detection tool powered by the [`piiranha-v1-detect-personal-information`](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information) model.

PII Hunter uses cutting-edge NLP and token classification to identify and annotate sensitive information like names, emails, phone numbers, addresses, SSNs, and more — directly from raw text.

---

## 🚀 Try it Instantly

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdityaBhatt3010/PII_Tryy/blob/main/PII_Classification.ipynb)

---

## 🧰 Features

* 🔍 Detect 15+ types of PII entities
* 🧠 Powered by Hugging Face Transformers
* ⚙️ GPU/CPU compatible
* 🧼 Outputs clean annotated text
* 🛡️ Ideal for redaction, sanitization, and compliance workflows

---

## 📦 Installation

```bash
pip install transformers
```

---

## 🧠 Model

* **Name:** `iiiorg/piiranha-v1-detect-personal-information`
* **Task:** Token Classification
* **Base:** HuggingFace Transformers (BERT-based)

---

## 🛠️ Usage

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

model_name = "iiiorg/piiranha-v1-detect-personal-information"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def annotate_pii(text):
    encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)
    pred_ids = torch.argmax(output.logits, dim=-1)[0].tolist()

    spans = []
    entity = None
    value, start = "", 0

    for i, (s, e) in enumerate(offsets):
        if s == e: continue
        label = model.config.id2label[pred_ids[i]]
        token = text[s:e]
        if label != "O":
            if entity is None:
                entity, value, start = label, token, s
            elif label == entity:
                value += token
            else:
                spans.append((start, value, entity))
                entity, value, start = label, token, s
        elif entity:
            spans.append((start, value, entity))
            entity, value = None, ""

    if entity:
        spans.append((start, value, entity))

    output = ""
    last = 0
    for s, v, l in sorted(spans):
        output += text[last:s]
        output += f"[{v} | {l}]"
        last = s + len(v)
    return output + text[last:]
```

---

## 📄 Example

```python
text = """
Aditya Bhatt sent a contract to client@securemail.com and called +1 (555) 123-4567 for confirmation. He also provided his SSN 123-45-6789.
"""

print(annotate_pii(text))
```

---

## ✅ Sample Output

```
[Aditya Bhatt | I-GIVENNAME] sent a contract to [client@securemail.com | I-EMAIL] and called [+1 (555) 123-4567 | I-TELEPHONENUM] for confirmation. He also provided his SSN [123-45-6789 | I-SOCIALNUM].
```

---

## 🧩 Supported Entity Types

* I-GIVENNAME, I-SURNAME
* I-EMAIL, I-TELEPHONENUM
* I-IDCARDNUM, I-ACCOUNTNUM
* I-CITY, I-ZIPCODE, I-STREET
* I-SOCIALNUM, I-COUNTRY
* ...and more

---

## 💼 Use Cases

* 🔐 PII Redaction
* 📜 Compliance (GDPR, HIPAA, etc.)
* 📊 Preprocessing for NLP Pipelines
* 🛡️ Log & Data Sanitization

---

## 👨‍💻 Author

**Aditya Bhatt**
Cybersecurity Professional | VAPT | TryHackMe Top 2% | Medium Writer
[GitHub](https://github.com/AdityaBhatt3010) | [LinkedIn](https://linkedin.com/in/aditya3010)

---

## 🧠 Credits

* [iiiorg/piiranha-v1](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information) for the PII detection model
* HuggingFace Transformers for model inference

---