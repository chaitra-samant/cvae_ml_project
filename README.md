# CVAE CelebA Project

This project trains a Conditional Variational Autoencoder (CVAE) on the CelebA dataset to generate faces based on specific attributes: **Eyeglasses**, **Smiling**, and **Mustache**.

## How to Run

Follow these steps to clone, set up, and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/cvae-celeba-project.git
cd cvae-celeba-project
```

(Replace `YOUR_USERNAME/cvae-celeba-project.git` with your repository's actual URL)

### 2. Set Up Your Python Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 3. Set Up Kaggle API

1. Go to https://kaggle.com/account  
2. Scroll to **API**  
3. Click **Create New API Token** (downloads `kaggle.json`)  
4. Place it here:

```
macOS/Linux: ~/.kaggle/kaggle.json
Windows:     C:\Users\<Your-Username>\.kaggle\kaggle.json
```

(Create the `.kaggle` folder if it doesn't exist)

### 4. Download the Dataset

```bash
python download_data.py
```

This downloads and extracts the CelebA dataset (~1.3GB).

### 5. Start Training

```bash
python main.py
```

You can also pass custom arguments:

```bash
python main.py --epochs 100 --lr 0.0005
```

### 6. Check Your Results

- Generated samples saved in: `samples_v4/`  
- Final model saved as: `cvae_eyeglasses_smiling_mustache.pth`
```
