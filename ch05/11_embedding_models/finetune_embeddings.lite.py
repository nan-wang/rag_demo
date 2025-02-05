import json
import random

import dotenv
from datasets import Dataset, DatasetDict
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers


def create_dataset_from_list(data_list, train_size=10, dev_size=2, test_size=2):
    random.shuffle(data_list)  # Shuffle the data to ensure random splits

    train_data = data_list[:train_size]
    dev_data = data_list[train_size : train_size + dev_size]
    test_data = data_list[train_size + dev_size : train_size + dev_size + test_size]

    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)
    test_dataset = Dataset.from_list(test_data)

    dataset_dict = DatasetDict(
        {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}
    )

    return dataset_dict


dotenv.load_dotenv()

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "jinaai/jina-embeddings-v3",
    trust_remote_code=True,
)
for param in model.parameters():
    param.requires_grad = True

# 3. Load a dataset to finetune on
with open("qa_triplets.json", "r") as f:
    json_list = json.load(f)
NUM_TRAIN = 14_000
NUM_EVAL = 3_000
NUM_TEST = 3_000
dataset = create_dataset_from_list(json_list, NUM_TRAIN, NUM_EVAL, NUM_TEST)
train_dataset = dataset["train"]
eval_dataset = dataset["dev"]
test_dataset = dataset["test"]

# 4. Define a loss function
loss = MultipleNegativesRankingLoss(model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/jina-embeddings-v3-olympics-v1",
    # Optional training parameters:
    num_train_epochs=10,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["user_query"],
    positives=eval_dataset["positive_document"],
    negatives=eval_dataset["negative_document"],
    name="olympics-dev",
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# (Optional) Evaluate the trained model on the test set
test_evaluator = TripletEvaluator(
    anchors=test_dataset["user_query"],
    positives=test_dataset["positive_document"],
    negatives=test_dataset["negative_document"],
    name="olympics-test",
)
test_evaluator(model)

# 8. Save the trained model
model.save_pretrained("models/jina-embeddings-v3-olympics/v1")
