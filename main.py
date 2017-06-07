# Employing a pretrained network for person re-identification problem.
# Evaluated on VIPeR and CUHK02 datasets.
# By: Bahram Lavi
# Email: bahram_lavi@yahoo.com


import sys;
sys.path.insert(0,"/path/to/your/caffe/python/") # Set your caffe path here!
import caffe
import numpy as np
import os

import matplotlib.pyplot as plt

def load_Caffe(len):
    #set this on your own path.
    net = caffe.Net('../caffe/models/bvlc_reference_caffenet/deploy.prototxt',
                    '../caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    caffe.TEST)
    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('../caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    net.blobs['data'].reshape(len, 3, 227, 227)

    return net,transformer

def viper_transformer():
    num_people=632
    viper_base_path='../Datasets/VIPeR/VIPeR/' # Set VIPeR path
    temps_path=viper_base_path+'cam_b/'
    probs_path=viper_base_path+'cam_a/'

    net, transformer = load_Caffe(num_people*2)

    list_temps_imgs_fileName = os.listdir(temps_path)
    list_probs_omgs_fileName=os.listdir(probs_path)

    im_lbl=0
    for img_prob in list_probs_omgs_fileName:
        print 'transforming for probe image #'+ (im_lbl+1).__str__()
        im=caffe.io.load_image(probs_path + img_prob)
        net.blobs['data'].data[im_lbl,...]=transformer.preprocess('data',im)
        im_lbl+=1
    for img_temp in list_temps_imgs_fileName:
        print 'transforming for template image #'+ (im_lbl+1).__str__()
        im=caffe.io.load_image(temps_path + img_temp)
        net.blobs['data'].data[im_lbl,...]=transformer.preprocess('data',im)
        im_lbl+=1

    net.forward()

    all=net.blobs['fc7'].data

    probes_data=all[0:num_people,:]
    temps_data=all[num_people:im_lbl,:]

    return temps_data,probes_data

def cuhk02_transformer():
    import os
    import h5py,pickle

    num_people=1816
    imgs_cam_a = []
    imgs_cam_b = []
    lbls_cam_a = []
    lbls_cam_b = []
    if os.path.isfile('./data/chuk02_temps_probs.pickle'):
        f = open('./data/cuhk02_temps_probs.pickle', 'r')
        temps, probs = pickle.load(f)
        f.close()
    else:
        if os.path.isfile('./data/chuk02_images.h5'):
            with h5py.File('./data/chuk02_images.h5', 'r') as hf:
                imgs_cam_a = hf['cam_a'][:]
                imgs_cam_b = hf['cam_b'][:]
                lbls_cam_a = hf['lbls_cam_a'][:]
                lbls_cam_b = hf['lbls_cam_b'][:]

        else:
            ds_path='/CUHK/CUHK02/Dataset/'     #set cuhk02 path
            pairs_ds=[ x for x in os.listdir(ds_path) if os.path.isdir(ds_path+x)]
            path_cam_a='/cam1'
            path_cam_b = '/cam2'

            all_imgs_count=0
            import glob
            for p in pairs_ds:
                list_imgs_cam_a=glob.glob(ds_path+p+path_cam_a+'/*.png')
                list_imgs_cam_b = glob.glob(ds_path+p+path_cam_b+'/*.png')
                all_imgs_count+=len(list_imgs_cam_a)
                for im_p in list_imgs_cam_a:
                    im=caffe.io.load_image(im_p)
                    imgs_cam_a.append(np.array(im))
                for im_p in list_imgs_cam_b:
                    im =caffe.io.load_image(im_p)
                    imgs_cam_b.append(np.array(im))
            import itertools
            lst = range(0, all_imgs_count/2)
            lbls_cam_a=list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in lst))
            lbls_cam_b=lbls_cam_a

            with h5py.File('./data/chuk02_images.h5', 'w') as hf:
                hf.create_dataset("cam_a", data=imgs_cam_a)
                hf.create_dataset("cam_b", data=imgs_cam_b)
                hf.create_dataset("lbls_cam_a", data=lbls_cam_a)
                hf.create_dataset("lbls_cam_b", data=lbls_cam_b)

        imgs_all=[]
        imgs_lbl=[]
        imgs_all.extend(imgs_cam_a)
        imgs_all.extend(imgs_cam_b)
        imgs_lbl.extend(lbls_cam_a)
        imgs_lbl.extend(lbls_cam_a)

        imgs_all = np.array(imgs_all)
        imgs_all = imgs_all.astype('float32')

        imgs_lbl = np.array(imgs_lbl)

        digit_indices = [np.where(imgs_lbl == i)[0] for i in range(0,1816)]
        temps,probs = create_pairs(imgs_all, digit_indices)

        import pickle
        f=open('./data/cuhk02_temps_probs.pickle','w')
        pickle.dump([temps,probs],f)
        f.close()


    if os.path.isfile('./data/cuhk02_tranformed_temps_probs.pickle'):
        f=open('./data/cuhk02_tranformed_temps_probs.pickle','r')
        probes_data,temps_data=pickle.load(f)
        f.close()

    else:
        net, transformer = load_Caffe(len(temps)*2)

        im_lbl=0
        for p in probs:
            print 'transforming for probe image #'+ (im_lbl+1).__str__()
            net.blobs['data'].data[im_lbl,...]=transformer.preprocess('data',p)
            im_lbl+=1
        for t in temps:
            print 'transforming for template image #'+ (im_lbl+1).__str__()
            net.blobs['data'].data[im_lbl,...]=transformer.preprocess('data',t)
            im_lbl+=1

        net.forward()

        all = net.blobs['fc7'].data

        probes_data = all[0:num_people, :]
        temps_data = all[num_people:len(all), :]

        with open('./data/cuhk02_tranformed_temps_probs.pickle','w') as f:
            pickle.dump([probes_data,temps_data],f)

    return temps_data, probes_data
def create_pairs(x,digit_indices):
    import random
    temps = []
    probs=[]
    n = min([len(digit_indices[d]) for d in range(len(digit_indices))])-1

    for d in range(len(digit_indices)):
        lbl = []
        r1 = random.randrange(0, n)
        r2 = random.randrange(0, n)
        z1, z2 = digit_indices[d][r1], digit_indices[d][r2]
        temps.append(x[z1])
        probs.append(x[z2])


    return np.array(temps),np.array(probs)

def accuracy_euclidean(temps,probes):
    from sklearn.metrics.pairwise import euclidean_distances

    all_scores = []
    num_p = len(probes)
    for i in range(num_p):
        print 'brobe #' + i.__str__()
        p = probes[i]
        s = euclidean_distances(p, temps)
        scores = s[0]

        all_scores.append(scores)

    return all_scores

def viper_partitions():
    import scipy.io as sio
    f=sio.loadmat('viperIDs.mat')
    return f['viper_ids_allRuns']

def CMC_viper(scores):
    inds= viper_partitions()
    all_rep=10
    final_cmc=[]
    for rep in xrange(all_rep):
        sc=[scores[i][inds[:,rep]] for i in range(len(scores))]
        sc=np.array(sc)
        sc=sc[inds[:,rep],:]
        cmc_rep=[]
        len_sc = len(sc)
        rank = []
        for i in range(len_sc):
            idx = np.argsort(sc[i])
            rank.append(np.where(idx == i)[0])
        rank_val = 0
        for r in xrange(len_sc):
            rank_val = rank_val + len([j for j in rank if r == j])
            cmc_rep.append(rank_val / float(len_sc))

        final_cmc.append(cmc_rep)

    final_cmc=np.average(final_cmc,axis=0)
    return final_cmc

def CMC_cuhk02(scores):
    all_rep = 10
    cmc_rep = []
    rank=[]
    len_sc = len(scores)

    for i in range(len_sc):
        idx = np.argsort(scores[i])
        rank.append(np.where(idx == i)[0])
    rank_val = 0
    for r in xrange(len_sc):
        rank_val = rank_val + len([j for j in rank if r == j])
        cmc_rep.append(rank_val / float(len_sc))


    return cmc_rep

def run_on_viper():
    temps, probs = viper_transformer()

    print 'Computing the similarity scores'
    similairities = accuracy_euclidean(temps, probs)

    print 'Computing the CMC'
    cmc = CMC_viper(similairities)

    plt.plot(cmc)
    plt.show()

def run_on_cuhk02():
    temps,probs=cuhk02_transformer()
    scores=accuracy_euclidean(temps,probs)
    cmc=CMC_cuhk02(scores)
    plt.plot(cmc)
    plt.show()
if __name__=='__main__':

    # run_on_viper()
    
    run_on_cuhk02()





