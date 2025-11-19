import kagglehub
import os

def download_celeba():
    dataset_path = kagglehub.dataset_download('jessicali9530/celeba-dataset')
    
    print(f"Dataset downloaded and extracted to: {dataset_path}")
    return dataset_path

if __name__ == "__main__":
    data_path = download_celeba()
    print(f"Dataset is located at: {data_path}")