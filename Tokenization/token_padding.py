from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

def tokenizer_padding(dataset_dict, checkpoint, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_dataset = dataset_dict.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_dataloader = DataLoader(
        tokenized_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    validation_dataloader = DataLoader(
        tokenized_dataset["validation"], batch_size=batch_size, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator
    )

    return train_dataloader, validation_dataloader, test_dataloader
