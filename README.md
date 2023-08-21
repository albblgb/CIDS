# CIDS
The official code for 'On Cover Independent Deep Neural Network Steganography'
    
#### List of reproduced papers
- [Here](https://github.com/albblgb/pusnet) provides the reproduced codes of HiDDeN, BalujaNet, wengNet, UDH, and HiNet 


## Dependencies and Installation
- Python 3.8.13, PyTorch = 1.11.0
- Run the following commands in your terminal:

  `conda env create  -f env.yml`

   `conda activate NIPS`


## Get Started
#### Dataset
- Generating cover image set
Run `python stable_diffusion.py`

- training dataset: [DIV2K](https://opendatalab.com/DIV2K) training dataset
- testing datasets:
1. [DIV2K](https://drive.google.com/file/d/1NYVWZXe0AjxdI5vuI2gF6_2hwoS1c4y7/view?usp=sharing) testing dataset
2. [1000 images randomly selected from the COCO dataset](https://drive.google.com/file/d/1NYVWZXe0AjxdI5vuI2gF6_2hwoS1c4y7/view?usp=sharing) 
3. 1000 images randomly selected from the CelebA dataset

#### Training
1. Change the code in `config.py`

    `line 4:  mode = 'train' ` 

2. Run `python CIDS.py`

#### Testing
1. Change the code in `config.py`

    `line4:  mode = 'test' `
  
    `line 41:  test_cids_path = '' `

2. Run `python CIDS.py`

- Here we provide [trained models](https://drive.google.com/drive/folders/1lM9ED7uzWYeznXSWKg4mgf7Xc7wjjm8Q?usp=sharing).
- The processed images, such as stego image and recovered secret image, will be saved at 'results/images'
- The training or testing log will be saved at 'results/cids_trained_on_div2k.log'


## Citation
If you find our paper or code useful for your research, please cite:
```

```
