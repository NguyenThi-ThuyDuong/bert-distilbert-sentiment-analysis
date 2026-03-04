# ==========================================================
# IMPORT THƯ VIỆN
# ==========================================================

import os
import time
import torch
import pandas as pd
import gradio as gr
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score

# ==========================================================
# LOAD CLEAN DATASET
# ==========================================================

file_path = "final_3_sentiment_clean_balanced.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("❌ Không tìm thấy file final_3_sentiment_clean_balanced.csv")

df = pd.read_csv(file_path)

df.rename(columns={
    "Sentence_vi": "text",
    "label_id": "label"
}, inplace=True)

df = df[["text", "label"]]

label_map = {
    0: "Tiêu cực",
    1: "Trung lập",
    2: "Tích cực"
}

# ==========================================================
# CHIA DATASET
# ==========================================================

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

total_data = len(df)
train_size = len(train_dataset)
test_size = len(test_dataset)

# ==========================================================
# TRAIN MODEL (chỉ chạy khi chưa có model)
# ==========================================================

def train_model(model_type):

    if model_type == "bert":
        model_name = "bert-base-multilingual-cased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
        save_path = "./bert_model"
    else:
        model_name = "distilbert-base-multilingual-cased"
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
        save_path = "./distil_model"

    def tokenize(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=48
        )

    encoded_train = train_dataset.map(tokenize, batched=True)
    encoded_test = test_dataset.map(tokenize, batched=True)

    encoded_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    encoded_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        num_train_epochs=2,
        save_strategy="no",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train
    )

    print(f"\n🚀 Train {model_type.upper()}...\n")
    trainer.train()

    preds = trainer.predict(encoded_test)
    y_pred = preds.predictions.argmax(axis=1)
    y_true = preds.label_ids

    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Accuracy {model_type.upper()}: {acc:.4f}")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return model, tokenizer, acc

# ==========================================================
# LOAD HOẶC TRAIN
# ==========================================================

if os.path.exists("./bert_model"):
    print("📂 Load BERT đã train")
    bert_tokenizer = BertTokenizer.from_pretrained("./bert_model")
    bert_model = BertForSequenceClassification.from_pretrained("./bert_model")
    bert_acc = 0.7491   # 👉 ghi số của bạn vào đây
else:
    bert_model, bert_tokenizer, bert_acc = train_model("bert")

if os.path.exists("./distil_model"):
    print("📂 Load DistilBERT đã train")
    distil_tokenizer = DistilBertTokenizer.from_pretrained("./distil_model")
    distil_model = DistilBertForSequenceClassification.from_pretrained("./distil_model")
    distil_acc = 0.7437   # 👉 ghi số của bạn vào đây
else:
    distil_model, distil_tokenizer, distil_acc = train_model("distil")

# ==========================================================
# PREDICT
# ==========================================================

def predict(text, model, tokenizer):

    start = time.time()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=48
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()

    end = time.time()

    label = label_map[pred_id]
    confidence = probs[0][pred_id].item() * 100
    inference_time = end - start

    return label, confidence, inference_time

# ==========================================================
# SO SÁNH MODEL
# ==========================================================

def compare_models(text):

    bert_label, bert_conf, bert_time = predict(text, bert_model, bert_tokenizer)
    distil_label, distil_conf, distil_time = predict(text, distil_model, distil_tokenizer)

    faster = "BERT nhanh hơn" if bert_time < distil_time else "DistilBERT nhanh hơn"

    return (
        bert_label, f"{bert_conf:.2f}%", f"{bert_time:.5f} giây",
        distil_label, f"{distil_conf:.2f}%", f"{distil_time:.5f} giây",
        faster
    )

# ==========================================================
# GIAO DIỆN
# ==========================================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(f"""
# 🔥 SO SÁNH BERT vs DistilBERT

### 📊 Thông tin dữ liệu
- Tổng dữ liệu: {total_data}
- Train size: {train_size}
- Test size: {test_size}

---

### 🎯 Accuracy Model
- 🧠 BERT: {bert_acc:.4f}
- ⚡ DistilBERT: {distil_acc:.4f}
""")

    text_input = gr.Textbox(label="Nhập câu cần phân tích", lines=2)
    analyze_btn = gr.Button("Phân tích", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🧠 BERT")
            bert_label_out = gr.Textbox(label="Nhãn")
            bert_conf_out = gr.Textbox(label="Độ tin cậy")
            bert_time_out = gr.Textbox(label="Thời gian")

        with gr.Column():
            gr.Markdown("## ⚡ DistilBERT")
            distil_label_out = gr.Textbox(label="Nhãn")
            distil_conf_out = gr.Textbox(label="Độ tin cậy")
            distil_time_out = gr.Textbox(label="Thời gian")

    faster_out = gr.Textbox(label="Model nhanh hơn")

    analyze_btn.click(
        compare_models,
        inputs=text_input,
        outputs=[
            bert_label_out,
            bert_conf_out,
            bert_time_out,
            distil_label_out,
            distil_conf_out,
            distil_time_out,
            faster_out
        ]
    )

demo.launch()