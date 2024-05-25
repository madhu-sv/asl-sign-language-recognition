# src/download_data.py

import os
import requests
import zipfile

def download_and_extract(url, extract_to='data'):
    """
    Download and extract a ZIP file.
    
    Args:
        url (str): URL of the ZIP file to download.
        extract_to (str): Directory to extract the contents to.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Download the file
    local_filename = os.path.join(extract_to, 'asl_data.zip')
    print(f"Downloading {url} to {local_filename}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

    # Extract the file
    print(f"Extracting {local_filename} to {extract_to}")
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

    # Remove the zip file
    os.remove(local_filename)
    print("Zip file removed.")

if __name__ == "__main__":
    # Example URL of the ASL dataset zip file
    dataset_url = "https://www.kaggle.com/datasets/grassknoted/asl-alphabet/asl_alphabet.zip"
    download_and_extract(dataset_url)
