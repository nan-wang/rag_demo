import json
import random

import dotenv
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

dotenv.load_dotenv()

from datasets import Dataset, DatasetDict


def create_dataset_from_list(data_list, train_size=10, dev_size=2, test_size=2):
    """
    Converts a list of dictionaries into a datasets DatasetDict with train, dev, and test splits.

    Args:
        data_list: A list of dictionaries, where each dictionary has keys "anchor", "positive", and "negative".
        train_size: The number of samples in the training split.
        dev_size: The number of samples in the development split.
        test_size: The number of samples in the test split.

    Returns:
        A datasets DatasetDict containing the train, dev, and test splits.
    """

    random.shuffle(data_list)  # Shuffle the data to ensure random splits

    train_data = data_list[:train_size]
    dev_data = data_list[train_size:train_size + dev_size]
    test_data = data_list[train_size + dev_size:train_size + dev_size + test_size]

    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)
    test_dataset = Dataset.from_list(test_data)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset
    })

    return dataset_dict


def main():
    # load the json file
    with open("qa_triplets.merged.json", "r") as f:
        data_list = json.load(f)
    num_train = 12_000
    num_eval = 3_000
    num_test = 3_000
    dataset = create_dataset_from_list(data_list, num_train, num_eval, num_test)
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    test_dataset = dataset["test"]
    print("First training example:", dataset["train"][0])
    print("First dev example:", dataset["dev"][0])
    print("First test example:", dataset["test"][0])

    # 1. Load a model to finetune with 2. (Optional) model card data
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v3",
        trust_remote_code=True,
    )
    for param in model.parameters():
        param.requires_grad = True

    # 3. Load a dataset to finetune on

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
        run_name="jina-embeddings-v3-olympics-0204v1",  # Will be used in W&B if `wandb` is installed
    )

    # 6. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
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
        anchors=test_dataset["anchor"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negative"],
        name="olympics-test",
    )
    test_evaluator(model)

    # 8. Save the trained model
    model.save_pretrained("models/jina-embeddings-v3-olympics/v1")

    # 9. (Optional) Push it to the Hugging Face Hub
    # model.push_to_hub("mpnet-base-all-nli-triplet")

    # CUDA_VISIBLE_DEVICES=0 HF_HOME=/home/jinaai/nanw/finetune_je_v3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python finetune_embeddings.py


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=3,4,5,6,7 HF_HOME=/home/jinaai/nanw/finetune_je_v3 accelerate launch --num_processes 4 finetune_embeddings.py
    # CUDA_VISIBLE_DEVICES=0 HF_HOME=/home/jinaai/nanw/finetune_je_v3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python finetune_embeddings.py
    main()
