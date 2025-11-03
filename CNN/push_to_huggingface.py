"""
Script to read data_all_images and push to HuggingFace Hub without processing.
"""
import os
from pathlib import Path
from PIL import Image
from datasets import Dataset, DatasetDict, Image, Features, Value
from huggingface_hub import login, HfApi
import glob


def read_and_push_to_hub(
    data_dir: str = "./data_all_images",
    repo_id: str = None,  # e.g., "username/dataset-name"
    private: bool = False,
    token: str = None
):
    """
    Read all images from data_all_images directory and push to HuggingFace Hub.
    
    Args:
        data_dir: Path to the directory containing images
        repo_id: HuggingFace Hub repository ID (e.g., "username/dataset-name")
        private: Whether to make the repository private
    """
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory {data_dir} does not exist")
    
    # Get all PNG images
    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    if len(image_paths) == 0:
        raise ValueError(f"No PNG images found in {data_dir}")
    
    print(f"Found {len(image_paths)} images")
    
    # Get image filenames (without extension for metadata)
    image_files = [os.path.basename(path) for path in image_paths]
    
    # Create dataset with images
    # Using Image feature to handle image data
    dataset_dict = {
        "image": [path for path in image_paths],
        "image_id": image_files
    }
    
    # Define features properly using Features class
    features = Features({
        "image": Image(),
        "image_id": Value("string")
    })
    
    dataset = Dataset.from_dict(dataset_dict, features=features)
    
    print(f"Created dataset with {len(dataset)} examples")
    print(f"Dataset features: {dataset.features}")
    
    # Push to Hub
    if repo_id:
        print(f"Pushing to HuggingFace Hub: {repo_id}")
        # Create the dataset repository first if it doesn't exist
        try:
            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
            print(f"Repository {repo_id} created/verified")
        except Exception as e:
            print(f"Note: Could not create repository (may already exist): {e}")
        
        # Push the dataset
        dataset.push_to_hub(repo_id, private=private, token=token)
        print(f"Successfully pushed to {repo_id}")
    else:
        print("No repo_id provided. Dataset created but not pushed to Hub.")
        print("To push, provide a repo_id parameter or set it in the script.")
        return dataset
    
    return dataset


if __name__ == "__main__":
    # Login to HuggingFace Hub with the write token
    login(token=HF_WRITE_TOKEN)
    print("Logged in to HuggingFace Hub")
    
    # Configure your dataset repository ID here
    REPO_ID = "ssuresh/idc-patches"  # Set this to your desired repo_id, e.g., "username/dataset-name"
    
    # Set to True if you want the repository to be private
    PRIVATE = False
    
    # Path to data directory (relative to script location)
    DATA_DIR = "./data_all_images"
    
    # Read and push to hub
    dataset = read_and_push_to_hub(
        data_dir=DATA_DIR,
        repo_id=REPO_ID,
        private=PRIVATE,
        token=HF_WRITE_TOKEN
    )


"""
Script to read data_all_images and push to HuggingFace Hub without processing.
"""
import os
from pathlib import Path
from PIL import Image
from datasets import Dataset, DatasetDict, Image, Features, Value
from huggingface_hub import login, HfApi
import glob

HF_WRITE_TOKEN = "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"

def read_and_push_to_hub(
    data_dir: str = "./data_all_images",
    repo_id: str = None,  # e.g., "username/dataset-name"
    private: bool = False,
    token: str = None
):
    """
    Read all images from data_all_images directory and push to HuggingFace Hub.
    
    Args:
        data_dir: Path to the directory containing images
        repo_id: HuggingFace Hub repository ID (e.g., "username/dataset-name")
        private: Whether to make the repository private
    """
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory {data_dir} does not exist")
    
    # Get all PNG images
    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    if len(image_paths) == 0:
        raise ValueError(f"No PNG images found in {data_dir}")
    
    print(f"Found {len(image_paths)} images")
    
    # Get image filenames (without extension for metadata)
    image_files = [os.path.basename(path) for path in image_paths]
    
    # Create dataset with images
    # Using Image feature to handle image data
    dataset_dict = {
        "image": [path for path in image_paths],
        "image_id": image_files
    }
    
    # Define features properly using Features class
    features = Features({
        "image": Image(),
        "image_id": Value("string")
    })
    
    dataset = Dataset.from_dict(dataset_dict, features=features)
    
    print(f"Created dataset with {len(dataset)} examples")
    print(f"Dataset features: {dataset.features}")
    
    # Push to Hub
    if repo_id:
        print(f"Pushing to HuggingFace Hub: {repo_id}")
        # Create the dataset repository first if it doesn't exist
        try:
            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
            print(f"Repository {repo_id} created/verified")
        except Exception as e:
            print(f"Note: Could not create repository (may already exist): {e}")
        
        # Push the dataset
        dataset.push_to_hub(repo_id, private=private, token=token)
        print(f"Successfully pushed to {repo_id}")
    else:
        print("No repo_id provided. Dataset created but not pushed to Hub.")
        print("To push, provide a repo_id parameter or set it in the script.")
        return dataset
    
    return dataset


if __name__ == "__main__":
    # Login to HuggingFace Hub with the write token
    login(token=HF_WRITE_TOKEN)
    print("Logged in to HuggingFace Hub")
    
    # Configure your dataset repository ID here
    REPO_ID = "ssuresh/idc-patches"  # Set this to your desired repo_id, e.g., "username/dataset-name"
    
    # Set to True if you want the repository to be private
    PRIVATE = False
    
    # Path to data directory (relative to script location)
    DATA_DIR = "./data_all_images"
    
    # Read and push to hub
    dataset = read_and_push_to_hub(
        data_dir=DATA_DIR,
        repo_id=REPO_ID,
        private=PRIVATE,
        token=HF_WRITE_TOKEN
    )

