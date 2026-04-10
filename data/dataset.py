import os
import zipfile
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ML_100K_URL = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_PATH = os.path.join(DATA_DIR, "ml-latest-small.zip")
EXTRACT_PATH = os.path.join(DATA_DIR, "ml-latest-small")

def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(EXTRACT_PATH):
        print("Downloading MovieLens dataset...")
        urllib.request.urlretrieve(ML_100K_URL, ZIP_PATH)
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(ZIP_PATH)
        print("Download and extraction complete.")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_and_extract()
