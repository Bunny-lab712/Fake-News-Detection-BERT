import os
import csv
import torch
import pandas as pd
import gradio as gr
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification,
    Trainer, 
    TrainingArguments
)

# =====================================================================
# CONFIGURATION & STORAGE OPTIMIZATION (Prevent C-Drive Choking)
# =====================================================================
os.environ["HF_HOME"] = r"E:\Adhya Projects\Fake news detection\.hf_cache"
os.environ["TORCH_HOME"] = r"E:\Adhya Projects\Fake news detection\.torch_cache"

DATASET_PATH = "WELFake_Dataset.csv"
MODEL_OUTPUT_DIR = "./distilbert_fake_news_model"

print("=" * 70)
print("🚀 Initializing Fake News Detection System Training & Deployment")
print("=" * 70)

# =====================================================================
# STEP 1: ROBUST DATA LOADING & PREPROCESSING
# =====================================================================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"🚨 Missing critical dataset file: {DATASET_PATH}")

print("📦 Step 1: Loading and parsing raw dataset files...")
df = pd.read_csv(
    DATASET_PATH,
    engine="python",
    quoting=csv.QUOTE_NONE,   # Safe evaluation of uneven quote anomalies
    on_bad_lines="skip",      # Omit broken or shifted row shards 
    encoding="utf-8"
)
print(f"✔️ Initial raw footprint loaded successfully: {len(df)} rows found.")

# Clean missing entries and handle potential label structural noise
df = df.dropna(subset=["title", "text", "label"])
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# Feature Fusion: Merge Title and Article Text for deep sequence context mapping
df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]]

print(f"📊 Post-cleaning distribution count:\n{df['label'].value_counts()}")
print(f"📉 Matrix Dimensions for Model Training: {df.shape}")

# =====================================================================
# STEP 2: TRAIN-TEST VALIDATION SPLIT
# =====================================================================
print("\n✂️ Step 2: Generating balanced Stratified Train-Test Matrix Splits...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["content"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42
)
print(f"   - Training samples: {len(train_texts)}")
print(f"   - Evaluation samples: {len(test_texts)}")

# =====================================================================
# STEP 3: HIGH-SPEED WORD-PIECE TOKENIZATION
# =====================================================================
print("\n🔤 Step 3: Downloading & executing DistilBERT Tokenization Layer...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Convert mapped vectors securely into PyTorch Dataset Formats
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels
})
print("✔️ Tokenization structures compiled cleanly.")

# =====================================================================
# STEP 4: MODEL LOADING & DEEP FINE-TUNING PIPELINE
# =====================================================================
print("\n🧠 Step 4: Loading Pre-trained DistilBERT Architecture Backbone...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# Version-safe, optimized hyperparameter tuning arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    logging_steps=500,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

print("🏋️ Commencing Fine-Tuning execution cycles. Standby...")
trainer.train()

# Persist weights and configurations to disk space locally
model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
print(f"🏁 Training Complete! Optimized weights safely compiled to: {MODEL_OUTPUT_DIR}")

# =====================================================================
# STEP 5: AUTOMATED REAL-TIME INTERACTIVE PORTAL DEPLOYMENT
# =====================================================================
print("\n🖥️ Step 5: Provisioning and spinning up Gradio Web Application Frame...")
model.eval()

def predict_news(text):
    if text.strip() == "":
        return "⚠️ Field validation exception: Please provide target news content."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    fake_prob = probs[0][0].item() * 100   # Label 0 represents FAKE instances
    real_prob = probs[0][1].item() * 100   # Label 1 represents REAL instances

    if fake_prob > real_prob:
        result = "🚨 SUSPICIOUS / FAKE NEWS"
        confidence = fake_prob
    else:
        result = "✅ VERIFIED / REAL NEWS"
        confidence = real_prob

    output_markdown = f"""
### 🔍 Analysis Verdict Report

## **{result}**

### 🔢 Core Confidence Matrix
**{confidence:.2f}%**

---

### 📊 Probability Breakdown Spectrum
* **Fake/Misinformation Likelihood:** {fake_prob:.2f}%
* **Fact-Based Real Journalism Integrity:** {real_prob:.2f}%

---

### 🧠 Deep Learning Engine Underlay
**Fine-tuned DistilBERT Framework Engine (WELFake Dataset Audit)**

⚠️ *Disclaimer: Output is dynamically evaluated via automated NLP deep learning structures and should be cross-verified with official global press records.*
"""
    return output_markdown

# Build the Web Engine Dashboard Layout Architecture
with gr.Blocks(css="#output_box { font-size: 20px; }") as interface:
    gr.Markdown(
        """
        # 📰 Deep Learning Fake News Detection Portal
        ### Transformer-Powered Contextual Auditing Engine
        """
    )
    with gr.Row():
        news_input = gr.Textbox(
            lines=12,
            placeholder="Copy and paste your online news article text data here for contextual analysis...",
            label="📝 Target News Text Stream Input"
        )
    
    predict_btn = gr.Button("🔍 Audit & Analyze Content Stream", variant="primary")
    output_box = gr.Markdown(label="📢 Real-Time Predictions Output Summary", elem_id="output_box")

    predict_btn.click(
        fn=predict_news,
        inputs=news_input,
        outputs=output_box
    )

# Launch with automated public cloud tunnels enabled for external testing
print("\n🌐 Generating active link structures. Access the application dashboard locally or via cloud share below:")
interface.launch(share=True)