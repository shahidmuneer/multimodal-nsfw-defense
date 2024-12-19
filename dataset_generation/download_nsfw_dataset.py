from datasets import load_dataset

ds = load_dataset("rickRossie/bluemoon_roleplay_chat_data_300k_messages")

# Display information about the dataset
print(ds)

# Access specific splits of the ds
train_split = ds['train'] if 'train' in ds else None
test_split = ds['test'] if 'test' in ds else None

# Example: Viewing the first few rows of the train split

# Save the train split locally as a CSV file
if 'train' in ds:
    ds['train'].to_csv("bluemoon_train.csv")
    print("Train split saved as 'bluemoon_train.csv'.")