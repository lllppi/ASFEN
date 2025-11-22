# ASFEN: Aspect-aware Semantic Feature Enhanced Networks for Multimodal Aspect-based Sentiment Analysis
Code and datasets of our paper "Aspect-aware Semantic Feature Enhanced Networks for Multimodal Aspect-based Sentiment Analysis".

## Requirements

- Python==3.6(recommended to use a virtual environment such as [Conda](https://docs.conda.io/en/latest/miniconda.html))
- torch==1.10.0
- transformers==3.4.0
- cython==0.29.13
- nltk==3.5
- numpy==1.19.5

To install requirements, run `pip install -r requirements.txt`.

## Preparation

Prepare dataset with:

`python dataset/preprocess_data.py`

You can also directly use the processed data in the Tweets15_corenlp/Tweets17_corenlp folder.  The document can be downloaded at [this link](https://pan.baidu.com/s/1O0BK9Ma3dPJbxenBODVJ9g?pwd=ddsw).

## Training

To train the ASFEN model, run:

`sh run.sh`



