!pip install --quiet datasets transformers torch gradio scikit-learn pandas matplotlib

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
import numpy as np
import gradio as gr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support



dataset = load_dataset("imdb")
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print("Train shape:", train_df.shape)
print(train_df.head())

# Class balance visualization
train_df['label'].value_counts().plot(kind="bar", title="Class Distribution (0=Neg,1=Pos)")
plt.show()

from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="distilgpt2")
set_seed(42)

synthetic_texts, synthetic_labels = [], []

# Generate 200 synthetic reviews (100 pos + 100 neg)
for _ in range(100):
    g = generator("This movie was amazing", max_length=50, do_sample=True)[0]['generated_text']
    synthetic_texts.append(g)
    synthetic_labels.append(1)
for _ in range(100):
    g = generator("This movie was terrible", max_length=50, do_sample=True)[0]['generated_text']
    synthetic_texts.append(g)
    synthetic_labels.append(0)

print("Synthetic samples:", len(synthetic_texts))
print(synthetic_texts[:2])

texts = train_df['text'].tolist() + synthetic_texts
labels = train_df['label'].tolist() + synthetic_labels

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize all texts
encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True, max_length=256)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = IMDbDataset(encodings, labels)
test_dataset = IMDbDataset(test_encodings, test_df['label'].tolist())

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

!pip install --upgrade transformers

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def predict_sentiment(text):
    if not text.strip():
        return "Please enter some text!"
    result = sentiment_pipeline(text)[0]
    label = "Positive 😊" if result['label'] == "LABEL_1" else "Negative 😡"
    score = result['score']
    return f"{label} (Confidence: {score:.2f})"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type any text here..."),
    outputs="text",
    title="Enhanced Sentiment Analysis with GPT2 Augmentation",
    description="Fine-tuned on real + GPT2-generated reviews. Works well for arbitrary text like 'hi', 'okay', etc."
)

demo.launch(share=True)
