# IMPLEMENTATION of [TI-CNN: Convolutional Neural Networks for Fake News Detection](https://arxiv.org/pdf/1806.00749v1.pdf)

This document contains instructions to [rebuild preprocessed data](#preprocessing-original-data), [reproduce results](#reproducing-results) and [test our pre-trained hdf5 models](#using-pretrained-models). [TICNN_REPORT](https://github.com/coderjedi/Data-Mining-Assignment-TICNN/blob/main/TICNN-Implementation/TICNN_REPORT.pdf) contains the technical documentation of our implementation combined with description of the novel TI-CNN-TITLE-1000 model proposed and experimental results obtained.  


## [Preprocessing Original Data](#prepro)
To obtain preprocessed dataset from [original dataset](https://drive.google.com/open?id=0B3e3qZpPtccsMFo5bk9Ib3VCc2c), refer to [INITIAL\_PREPROCESSING/](https://github.com/coderjedi/Data-Mining-Assignment-TICNN/tree/main/TICNN-Implementation/INITIAL_PREPROCESSING) folder. [TICNN\_Preprocessing.ipynb](https://github.com/coderjedi/Data-Mining-Assignment-TICNN/blob/main/TICNN-Implementation/INITIAL_PREPROCESSING/TICNN_Preprocessing.ipynb) preprocesses the original dataset to save final\_text\_df.pkl (final dataframe to be used for text only models) and final\_image\_df.pkl (intermediary dataframe contains datapoints of which images were retrieved succesfully, along with explicit image features). [Pre-trained caffe model](https://github.com/vinuvish/Face-detection-with-OpenCV-and-deep-learning/blob/master/models/deploy.prototxt.txt) is used for obtaining image explicit attributes (refer to the report).   
[Preprocessing\_image\_files.ipynb](https://github.com/coderjedi/Data-Mining-Assignment-TICNN/blob/main/TICNN-Implementation/INITIAL_PREPROCESSING/Preprocessing_image_files.ipynb) preprocesses final\_image\_df.pkl to obtain df\_final\_new.pkl dataframe containing the preprocessed images ready to use in the models using the image data.
To skip these steps, kindly download these pickle files from following links and place these files in a folder named TICNN/ at outermost directory level for smooth functioning of the code.
* [final\_text\_df.pkl](https://drive.google.com/file/d/1urxltOuRs-wufLfZSvI5zdVZzc06Wtk-/view?usp=sharing)
* [final\_image\_df.pkl](https://drive.google.com/file/d/1grzlAGZk_IfniDPJWJObsBKPkgwH4Ny8/view?usp=sharing)
* [df\_final\_new.pkl](https://drive.google.com/file/d/1fXLXM_zfekyW6SrwlQLgEM3Jz0JM74Zh/view?usp=sharing)

## [Reproducing Results](#resu)
We present code for four models namely GRU-400, LSTM-400, CNN-Text-1000 and our novel TICNN-TITLE-1000 model which is improved version of TICNN-1000 model mentioned in original paper. To further see details of these models refer to documentation and presentations provided. [GRU-400 notebook](https://github.com/coderjedi/Data-Mining-Assignment-TICNN/blob/main/TICNN-Implementation/GRU-400/GRU-400.ipynb) , [CNN-text-1000 notebook](https://github.com/coderjedi/Data-Mining-Assignment-TICNN/blob/main/TICNN-Implementation/CNN-Text-1000/CNN_text_1000_training.ipynb), [LSTM-400 notebook](https://github.com/coderjedi/Data-Mining-Assignment-TICNN/blob/main/TICNN-Implementation/LSTM-400/Train_LSTM.ipynb), [TICNN-TITLE-1000 notebook](https://github.com/coderjedi/Data-Mining-Assignment-TICNN/blob/main/TICNN-Implementation/TICNN-TITLE-1000/TICNN_TITLE_1000_training.ipynb) contains training code for respective models. All models require publicly available [glove-100 file](https://drive.google.com/file/d/14CJFRKctq_lioE8FHst7Zhw259rquC25/view?usp=sharing) for the embedding layer. Preprocessed pickle files from previous section are also required depending on model's modality. Please place these files in your mounted google drive while running these collab notebooks. Hdf5 files for the pretrained models will be automatically saved, which can be used for inference. All the models have been cross-validated. To skip these steps, please use already trained hdf5 files provided [here](#pretr). For directory structure please refer to the drive [link](https://drive.google.com/drive/folders/1ORZu6amtwe_3bGScTUTc2hvZ7lM0V9Ft?usp=sharing). 


## [Using pretrained models](#pretr)

[GRU-400 notebook](https://github.com/coderjedi/Data-Mining-Assignment-TICNN/blob/main/TICNN-Implementation/GRU-400/GRU_400_test.ipynb) , [CNN-text-1000 notebook](https://github.com/coderjedi/Data-Mining-Assignment-TICNN/blob/main/TICNN-Implementation/CNN-Text-1000/CNN_Text_1000_test.ipynb), [LSTM-400 notebook](https://github.com/coderjedi/Data-Mining-Assignment-TICNN/blob/main/TICNN-Implementation/LSTM-400/LSTM_Test.ipynb), [TICNN-TITLE-1000 notebook](https://github.com/coderjedi/Data-Mining-Assignment-TICNN/blob/main/TICNN-Implementation/TICNN-TITLE-1000/TICNN_TITLE_1000_test_demo.ipynb) contains  code for inference on respective models on the test set.
If only inference is to be verified then following pretrained models can be used directly.
* [GRU-400 pretrained model](https://drive.google.com/file/d/1QNabVOOKnEe92tcoRFcE4lQhT3Lbeffu/view?usp=sharing) 
* [LSTM-400 pretrained model](https://drive.google.com/file/d/1G7JbUmf9pgTNKmI9vAkm2cZrq9_3Mktv/view?usp=sharing) 
* CNN-TEXT-1000 pretrained model [h5](https://drive.google.com/file/d/1f85SOgyCKMmyUYjX_FmoidX4j8pwWGcA/view?usp=sharing) and [json](https://drive.google.com/file/d/1bbekrCFJHBJNUI0BgcDuH1y0Tx7oVyLt/view?usp=sharing) files. 
* TICNN-TITLE-1000 pretrained model [h5](https://drive.google.com/file/d/1JHNQo882K6RsLA7l2yDooDzZzpzG8iFq/view?usp=sharing) and [json](https://drive.google.com/file/d/1aZizdNMuHd3eAzlnKHlk83nE8HHf9o3x/view?usp=sharing) files. 

### Credits
This project was done as partial requirement for Data Mining Course under Dr. Yashwardhan Sharma, Bits-Pilani, Pilani Campus. Contributors are:- Naman Goenka, Himanshu Pandey, Ayush Singh, Harshita Gupta
(All contributors have contrbuted equally).




