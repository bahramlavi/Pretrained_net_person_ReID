# Pretrained_net_person_ReID
This code employs a Covlotional Neural Network (CNN) model which provided in the Caffe library,<a href="https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet">BVLC Reference CaffeNet</a> , to generate image represation for person re-identification. In this manner, the network takes images of a re-id dataset as input, and return corresponding verctor representations as the output of the fully connected layer (fc7). 

The experimental results have be carried out on both VIPeR and CUHK02 datasets. 

Note that before running the code, make sure you set the pathes for caffe and the datasets, accordingly. 


