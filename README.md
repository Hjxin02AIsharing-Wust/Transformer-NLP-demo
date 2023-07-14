# Transformer-NLP-demo

This is our application based on the transformer network structure for the text summarization task. Here's [my blog](https://www.cnblogs.com/Hjxin02AIsharing-Wust/p/17547701.html) on some interpretations of the transformer paper.

The **transformer** paper 《[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)》. Here's a diagram of its network structure, which is an encoder-decoder structure, and we've unpacked each of its components in the [blog](https://www.cnblogs.com/Hjxin02AIsharing-Wust/p/17547701.html).

![image](https://github.com/Hjxin02AIsharing-Wust/Transformer-NLP-demo/blob/main/image_texture/transformer8.png).

## Dataset

Our text summary dataset is in the `data` folder of the project, it contains the following elements: Headline, Short, Source, Time, Publish Date. 


## Data Preprocess

We use the Short as data and Headline as the label, so we need to delete the other elements in the dataset. Text is used as serial input data to the decoder, to which we add an identifier before and after. In addition to encoding the text. You can see these in `Data_Preprocess.py`.

## Experiment

Please install the latest [Anaconda](https://www.anaconda.com/download/) distribution. We ran our experiments with Tensorflow 2.12.0, numpy 1.23.5, CUDA 11.1, Python 3.8.16 and Windows. We have successfully trained models with tensorflow.  Here are some hyperparameter settings for the transformer network structure, with the number of encoder and decoder layers set to 4, the vector dimension set to 128, the sequence dimension set to 512, and the number of multi-head attention set to 8. Adam optimizer is used to train 10 epochs on an RTX 1050Ti GPU. 

## Usage

### Train

You can use the following commands to train the model, the weights file for the model is kept in the `checkpoints` folder：
```shell
python train.py 
```

### Inference


Here you can select any text for your input, after that run this code `Inference.py` and you will see the summarized output.


