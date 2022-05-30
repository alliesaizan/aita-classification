import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer

# code taken from https://huggingface.co/docs/transformers/tasks/sequence_classification

# Load in posts
posts = pd.read_csv("data/raw/big-query-aita-aug18-aug19.csv")
posts = posts.loc[(posts["selftext"] != "[deleted]") & (posts["selftext"].isnull() == False)]
posts["a_hole"] = np.where(posts["link_flair_text"] == "Asshole", 1, 0)

# train-test-split and data prep for tokenization
train_posts, test_posts, train_label, test_label = train_test_split(posts[["selftext"]], posts[["a_hole"]], train_size = 0.75)
train_posts = train_posts["selftext"].tolist() 
test_posts = test_posts["selftext"].tolist()
train_label = train_label["a_hole"].tolist() 
test_label = test_label["a_hole"].tolist() 

# Preprocess data using HuggingFace's autotokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only = True)
train_encodings = tokenizer(train_posts, truncation=True)
test_encodings = tokenizer(test_posts, truncation=True)

# Save encodings in case we need them later!
# torch.save(train_encodings, "data/processed/train_encodings.pkl")
# torch.save(test_encodings, "data/processed/test_encodings.pkl")

# Create custom datasets
class customDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = customDataset(train_encodings, train_label)
test_dataset = customDataset(test_encodings, test_label)

# Define data collator and training arguments
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# train model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to('cuda:0')

import gc
#del train_encodings, test_encodings, posts
gc.collect()


training_args = TrainingArguments(
    output_dir="results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy = "epoch",
    save_strategy ="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
    )

torch.cuda.empty_cache()

trainer.train()

# trainer.evaluate()
# from transformers import pipeline
# pipe = pipeline("text-classification", model="results/TBD", device=0)
