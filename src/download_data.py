import kagglehub
import shutil
import os

def download_dataset():
    path = kagglehub.dataset_download("alfathterry/telco-customer-churn-11-1-3")

    os.makedirs("data", exist_ok=True)

    print("Downloaded files:", os.listdir(path))

    # Copy ONLY CSV file
    for file in os.listdir(path):
        if file.endswith(".csv"):
            shutil.copy(os.path.join(path, file), "data/")
            print("Copied:", file)

if __name__ == "__main__":
    download_dataset()