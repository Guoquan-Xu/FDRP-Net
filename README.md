# FDRP-Net

## Requirements
- Python 3.11 (recommended)
- PyTorch 2.4.0+cu118 (recommended)
- Additional packages listed in `requirements.txt`

## Training
To train FDRP-Net, please follow these steps:
1. Download the training dataset from Dataset_Link.
2. Configure the training data path in ModelWithColor.json.
3. Run the training script:
   ```python
   python train.py
   ```

## Testing
To test FDRP-Net, follow these steps:
1. Download test datasets: T89, [Challenge60](https://pan.baidu.com/s/1k2j7Ft1dWyAIpBkg2dTALA?pwd=ucgg), [U45](https://pan.baidu.com/s/1dTB40EqJ0GBfQHXOVDTQRA?pwd=8crw), [UIQS](https://pan.baidu.com/s/11FCB6tKJaApHPyGnl-SmVw?pwd=4qmw), [UFO](https://pan.baidu.com/s/1ahXBTinsK3EV_oogj4mvZQ?pwd=cqiw), [SUIM](https://pan.baidu.com/s/11UFX9oUuWehza4le7uLYCQ?pwd=ez2n), [EUVP-Scenes](https://pan.baidu.com/s/108djEoCIAQ5SJM-21HgN9Q?pwd=5rn4).
2. Download the pre-trained weights from [Weights](https://drive.google.com/drive/folders/1dKiMT7aE9Gfvg3oqfzd2xOxdKs56jTMh?usp=drive_link). Place `URanker_ckpt.pth` under the `./Norm/URanker` directory. Then, configure the path to the pre-trained weights file in `ModelWithColor.json`.
3. Configure the test data path in ModelWithColor.json. Optionally, set the result log file name in test.py.
4. Run the testing script:
   ```python
   python test.py --outImg <path/to/output_images>
   ```
