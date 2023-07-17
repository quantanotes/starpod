import requests
import os
from tqdm import tqdm

# Shamelessly copied from https://github.com/mzbac/GPTQ-for-LLaMa-API/blob/master/download.py <3

def download_file(url: str, path: str):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
            
    progress_bar.close()

def download(name: str):
    folder = os.path.join(os.getcwd(), "weights/")
    
    base_url = f"https://huggingface.co/{name}/resolve/main"
    headers = {"User-Agent": "Hugging Face Python"}
    response = requests.get(f"https://huggingface.co/api/models/{name}", headers=headers)
    response.raise_for_status()
    
    files = [file["rfilename"] for file in response.json()["siblings"]]

    os.makedirs(f"{folder}/{name}", exist_ok=True)

    for file in files:
        print(f"Downloading {file}...")
        download_file(f"{base_url}/{file}", f"{folder}/{name}/{file}")
