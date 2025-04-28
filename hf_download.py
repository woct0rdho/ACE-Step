from concurrent.futures import ThreadPoolExecutor
import json
import random
import time
import requests
import os
from tqdm import tqdm

# Create a session for requests
session = requests.Session()

def get_all_files(repo_name: str, headers, is_dataset=False, folder_name=None):
    """
    Retrieve all files information from a repository.
    If the local JSON and TXT files exist, read from them; otherwise, fetch from the API and save locally.

    :param repo_name: Name of the repository.
    :param is_dataset: Whether it's a dataset repository.
    :param folder_name: Optional folder name to filter files.
    :return: A list of dictionaries containing file information (filename and URL).
    """
    HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    # Check if the local JSON file exists
    if os.path.exists(f"repos/{repo_name}.json"):
        with open(f"repos/{repo_name}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # Determine the API URL based on whether it's a dataset or a model
        if is_dataset:
            url = f"{HF_ENDPOINT}/api/datasets/{repo_name}"
        else:
            url = f"{HF_ENDPOINT}/api/models/{repo_name}"
        response = session.get(url, headers=headers)
        data = response.json()
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(f"repos/{repo_name}.json"), exist_ok=True)
        with open(f"repos/{repo_name}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=4, ensure_ascii=False))
    all_files = []
    # Check if the local TXT file exists
    if os.path.exists(f"repos/{repo_name}.txt"):
        with open(f"repos/{repo_name}.txt", "r", encoding="utf-8") as f:
            for line in f:
                filename, url = line.strip().split("\t")
                if folder_name and folder_name not in filename:
                    continue
                all_files.append({"filename": filename, "url": url})
    else:
        if "siblings" in data:
            with open(f"repos/{repo_name}.txt", "w", encoding="utf-8") as f:
                for file in data["siblings"]:
                    if is_dataset:
                        url = f"https://huggingface.co/datasets/{repo_name}/resolve/main/{file['rfilename']}"
                    else:
                        url = f"https://huggingface.co/{repo_name}/resolve/main/{file['rfilename']}"
                    if folder_name and folder_name not in file['rfilename']:
                        continue
                    f.write(f"{file['rfilename']}\t{url}\n")
                    all_files.append({"filename": file['rfilename'], "url": url})
    return all_files


def download_file(url, file_name, save_path, headers):
    """
    Download a single file with resume support.

    :param url: URL of the file to download.
    :param file_name: Name of the file to save.
    :param save_path: Path to save the file.
    :param headers: Headers for the HTTP request.
    """
    file_path = f"{save_path}/{file_name}"
    resume_byte = 0
    # Check if the file already exists and get its size
    if os.path.exists(file_path):
        resume_byte = os.path.getsize(file_path)
        print(f"Resuming download for {file_name} from byte {resume_byte}")
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Copy headers and add Range header if resuming
    local_headers = headers.copy()
    if resume_byte > 0:
        local_headers['Range'] = f"bytes={resume_byte}-"
    with session.get(url, stream=True, headers=local_headers) as r:
        # Check if the file is already fully downloaded
        if r.status_code == 416:
            print(f"{file_name} is already fully downloaded")
            return
        r.raise_for_status()
        total_length = int(r.headers.get('content-length', 0)) + resume_byte
        with open(file_path, "ab") as f:
            with tqdm(total=total_length, initial=resume_byte, unit='B', unit_scale=True, unit_divisor=1024, desc=f"Downloading {file_name}") as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    print(f"Downloaded {file_name}")


def download_repo(repo: str, save_path: str, is_dataset=False, folder_name=None, retry=10):
    """
    Download all files from a repository.

    :param repo: Name of the repository.
    :param save_path: Path to save the downloaded files.
    :param is_dataset: Whether it's a dataset repository.
    :param folder_name: Optional folder name to filter files.
    :param retry: Number of retries in case of errors.
    """
    access_token_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")
    if not os.path.exists(access_token_path):
        raise FileNotFoundError(f"Access token file not found at {access_token_path}. Please log in to Hugging Face CLI.")
    with open(access_token_path, "r", encoding="utf-8") as f:
        access_token = f.read().strip()
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    retry_count = 0
    all_files = []
    while retry_count < retry:
        try:
            all_files = get_all_files(f"{repo}", headers, is_dataset, folder_name)
            break
        except Exception as e:
            print(f"Error fetching file list: {e}")
            retry_count += 1
            if retry_count == retry:
                print("Max retries reached. Exiting.")
                return
    for file_data in all_files:
        retry_count = 0
        while retry_count < retry:
            try:
                file_name = file_data["filename"]
                url = file_data["url"]
                print(f"Start Download {file_name}")
                download_file(url, file_name, save_path, headers)
                print(f"Downloaded {file_name}")
                break
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")
                retry_count += 1
                time.sleep(random.uniform(1, 5) * retry_count)
                if retry_count == retry:
                    print(f"Max retries reached for {file_name}. Exiting.")
                    break
    print("Download completed")


def download_part(url, start, end, save_path, file_name, headers, part_idx):
    """
    Download a part of a file.

    :param url: URL of the file to download.
    :param start: Start byte position of the part.
    :param end: End byte position of the part.
    :param save_path: Path to save the part.
    :param file_name: Name of the file.
    :param headers: Headers for the HTTP request.
    :param part_idx: Index of the part.
    :return: Path of the downloaded part.
    """
    part_path = os.path.join(save_path, f"{file_name}.part{part_idx}")
    resume_byte = 0
    if os.path.exists(part_path):
        resume_byte = os.path.getsize(part_path)
    if resume_byte > 0:
        start += resume_byte
    local_headers = headers.copy()
    local_headers['Range'] = f"bytes={start}-{end}"
    try:
        response = session.get(url, headers=local_headers, stream=True)
        response.raise_for_status()
        with open(part_path, 'ab' if resume_byte else 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except requests.RequestException as e:
        print(f"Error downloading part {part_idx}: {e}")
    return part_path


def combine_parts(file_parts, destination):
    """
    Combine downloaded parts into a single file.

    :param file_parts: List of paths of the downloaded parts.
    :param destination: Path of the combined file.
    """
    with open(destination, 'wb') as destination_file:
        for part in file_parts:
            with open(part, 'rb') as part_file:
                destination_file.write(part_file.read())
            os.remove(part)


def download_file_multi_part(url, file_name, save_path, headers, min_size=1024*1024*10, num_threads=4):
    """
    Download a file in multiple parts.

    :param url: URL of the file to download.
    :param file_name: Name of the file to save.
    :param save_path: Path to save the file.
    :param headers: Headers for the HTTP request.
    :param min_size: Minimum size to split the file.
    :param num_threads: Number of threads to use.
    """
    file_path = os.path.join(save_path, file_name)
    if os.path.exists(file_path):
        if file_path.endswith(".tar") and os.path.getsize(file_path) < 1024 * 1024:
            os.remove(file_path)
        else:
            print(f"{file_name} already exists")
            return
    response = session.head(url, headers=headers)
    file_size = int(response.headers.get('content-length', 0))
    if file_size == 0:
        raise Exception("Cannot get file size from server")
    print(f"Start Download {file_name} ({file_size} bytes)")
    if file_size <= min_size or num_threads == 1:
        file_parts = [download_part(url, 0, file_size - 1, save_path, file_name, headers, 0)]
    else:
        part_size = file_size // num_threads
        file_parts = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(download_part, url, i * part_size, (i + 1) * part_size - 1 if i < num_threads - 1 else file_size - 1, save_path, file_name, headers, i) for i in range(num_threads)]
            file_parts = [future.result() for future in futures]
    combine_parts(file_parts, file_path)
    print(f"Downloaded {file_name}")


def download_repo_multi_part(repo: str, save_path: str, headers):
    """
    Download all files from a repository in multiple parts.

    :param repo: Name of the repository.
    :param save_path: Path to save the downloaded files.
    :param headers: Headers for the HTTP request.
    """
    all_files = get_all_files(f"{repo}", headers)
    for file_data in all_files:
        file_name = file_data["filename"]
        url = file_data["url"]
        download_file_multi_part(url, file_name, save_path, headers, num_threads=8)
    print("Download completed")


if __name__ == "__main__":
    download_repo("timedomain/fusic_v1", "checkpoints_new")
