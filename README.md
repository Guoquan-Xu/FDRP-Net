# FDRP-Net

## Requirements
- Python 3.11 (recommended)
- PyTorch 2.4.0+cu118 (recommended)
- Additional packages listed in `requirements.txt`

## Training
To train FDRP-Net, please follow these steps:
1. Download the training dataset from [Dataset Link].
2. Configure the training data path in ModelWithColor.json.
3. Run the training script:
   ```python
   python train.py
   ```

## Testing
To test FDRP-Net, follow these steps:
1. Download test datasets: Challenge60, U45, UIQS, UFO, SUIM, EUVP-Scenes.
2. Download the pre-trained weights from [Weights Link].
3. Configure the test data path in ModelWithColor.json. Optionally, set the result log file name in test.py.
4. Run the testing script:
   ```python
   python test.py --outImg <path/to/output_images>
   ```

## model_files
You can obtain it through [model_files](https://drive.google.com/drive/folders/1dKiMT7aE9Gfvg3oqfzd2xOxdKs56jTMh?usp=drive_link)
