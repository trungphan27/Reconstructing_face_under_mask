import os

# Use local dataset path
path = os.path.join(os.path.dirname(__file__), "dataset")

print("Path to dataset files:", path)
print(f"\nDataset structure:")
print(f"- with_mask: {len(os.listdir(os.path.join(path, 'with_mask')))} images")
print(f"- without_mask: {len(os.listdir(os.path.join(path, 'without_mask')))} images")