import argparse
import sys
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate

def compute_metrics(eval_pred):
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    
    return {"accuracy": accuracy, "f1": f1}

def train_model(output_dir="emotion_model"):
    print("Loading dataset 'dair-ai/emotion'...")
    dataset = load_dataset("dair-ai/emotion")
    
    model_name = "distilbert-base-uncased"
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Small subset for demonstration/testing speed
    # Remove these lines for full training
    print("Selecting subset for demonstration (remove in production)...")
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(200))

    print(f"Loading model {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,  # 3 is standard, 1 for demo speed
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        use_cpu=False # Set to True if no GPU/MPS
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    
    print("Training complete.")
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_model()
