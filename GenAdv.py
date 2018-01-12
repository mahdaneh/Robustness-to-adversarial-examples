#
"""
compatible with python version 2.7
@author: mahdieh.abbasi.1@ulaval.ca
"""

from __future__ import print_function
from __future__ import division
import matplotlib
matplotlib.use ('Agg')
import sys
sys.setrecursionlimit(10000)
import os
import theano
import theano.tensor as T
import numpy as np
import lasagne
import cPickle as pickle
from scipy.optimize import fmin_l_bfgs_b as cnstOpt
import getopt
import load_data as ldb
import CNN_building as CNN
import Utility as util
import gzip


def Load_weights(Pretrained_net,Arch , dataset_type,BN):
    X_val = None
    y_val = None
    if dataset_type=='cifar10':
        print('load cifar10 dataset')
        normaliz_input = False

        size_images = 32
        num_channel = 3
        num_classes = 10
        X_train, y_train, X_test, y_test,  meanpixel = ldb.load_cifar_dataset()
        print('min of train image before Nor '+str ( np.min(X_train)))
        print('min of mean pixel'+str ( np.min(meanpixel)))

        X_train -=  meanpixel
        X_test -=  meanpixel

    elif dataset_type == 'mnist':
        print('load mnist dataset')
        normaliz_input = False

        size_images = 28
        num_channel = 1
        num_classes = 10
        X_train, y_train,  X_test, y_test = ldb.load_MNIST_dataset()
        meanpixel = np.zeros(X_train[0].shape)
    elif dataset_type =='SVHN':

        normaliz_input = True
        print('Street View is selected')
        size_images=32
        num_channel = 3
        num_classes = 10

        X_train, y_train, X_val, y_val, X_test, y_test, EignVectors, EignValues = ldb.load_SVNH_dataset( )
        meanpixel = np.zeros(X_train[0].shape)
    elif dataset_type == 'cifar100':
        print('load cifar100 dataset')

        normaliz_input = False
        size_images = 32
        num_channel = 3
        num_classes = 100
        X_train, y_train, X_test, y_test, meanpixel = ldb.load_cifar100_dataset(Normal_flag= True)
        print('number of training '+str(len(y_train)))

    clip_min = np.min(X_train)
    clip_max = np.max(X_train)
    print\
        ('clip min '+str(clip_min))
    print('clip max ' + str(clip_max))

    input_var = T.tensor4('inputs', dtype='float32')
    target_var = T.ivector('targets')

    if Arch =='NiN':
        # create CNN
        num_filter =[192,160,96,192,192,192,192,192,num_classes]
        print('select NiN structure')
        network  = CNN.NiN(input_var=input_var, num_chan= num_channel ,width = size_images,num_fill =num_filter)
    elif Arch =='cuda_conv':
        num_filter = [32, 32, 64]
        print('Batch Normalization'+str(BN))
        network , logit_layer = CNN.Cuda_Conv_18(input_var=input_var, num_chan=num_channel, width=size_images, num_fill=num_filter, FCparams=0, num_outputs=num_classes, flag=False, normaliz_input = normaliz_input, BN=BN)
    elif Arch == 'resnet':
        n=18
        network, logit_layer = CNN.ResNet_FullPreActivation(input_var=input_var, PIXELS=size_images, num_outputs=num_classes, n=n)
    elif Arch =='VGG':
        network, logit_layer = CNN.VGG2(type='VGG16', input_var=input_var, num_chan=num_channel, width=size_images,
                          num_outputs=num_classes)


    # get the softmax output
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    # get the logit vector of the net (the output before softmax activation function)
    logits = lasagne.layers.get_output(logit_layer, deterministic=True)

    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
    test_loss = test_loss.mean(axis=0)
    test_acc = T.eq(T.argmax(test_prediction, axis=1), target_var)

    # Theano functions for loss and logits output
    logits_fn = theano.function([input_var],[logits], allow_input_downcast=True)
    predic_fun = theano.function([input_var], [test_prediction], allow_input_downcast=True)
    val_fn_loss = theano.function([input_var, target_var], [test_loss], allow_input_downcast=True)
    val_fn_acc = theano.function([input_var, target_var], [test_acc],allow_input_downcast=True)

    # gradient of the loss respect to input
    grad_loss = theano.grad(test_loss, input_var)
    fun_grad_loss = theano.function([input_var, target_var], grad_loss, allow_input_downcast=True)

    # the gradient of logit of each class respect to input used for DeepFool attack
    [dfdx, Pert] = theano.scan(lambda j, logits , input_var : T.grad(logits[0,j], input_var), sequences=T.arange(num_classes),
                             non_sequences=[logits,input_var])
    fun_grad_classifir = theano.function([input_var],[dfdx])


    print('load pretrained weights from '+str(Pretrained_net))
    # load the trained weights of CNN from a .pkl file
    try:
        net = pickle.load(open(Pretrained_net, 'r'))
    except:
        net = pickle.load(gzip.open(Pretrained_net, 'r'))


    # the loaded network (i.e. net) is a dictionary, where net['params'] is the trained weights of the loaded network
    all_param = net['params']
    lasagne.layers.set_all_param_values(network, all_param)

    # compute the accuracy on training set and test set. For generating adversaries only for correctly classified samples
    all_acc = []
    for batch in util.iterate_minibatches(X_train, y_train, 100, shuffle=False):
        inputs, targets, indices = batch
        all_acc.append(val_fn_acc(inputs,targets))
    all_acc = np.asarray(np.hstack(all_acc))
    indx_corrc = np.where(all_acc[0]==1)
    print('accuracy on training  '+str(np.mean(all_acc[0])))
    pickle.dump(indx_corrc,open(dataset_type+'_Index_correcltyClassifiedTrain_'+Arch+'.pkl','wb'))

    all_acc = []
    for batch in util.iterate_minibatches(X_test, y_test, 100, shuffle=False):
        inputs, targets, indices = batch
        all_acc.append(val_fn_acc(inputs,targets))
    all_acc = np.asarray(np.hstack(all_acc))
    indx_corrc = np.where(all_acc[0]==1)
    counting = np.asarray([len(np.where(y_test[indx_corrc] == i)[0]) for i in range(num_classes)])
    print('accuracy on test  '+str(np.mean(all_acc[0])))
    pickle.dump(indx_corrc,open(dataset_type+'_Index_correcltyClassifiedTest_'+Arch+'.pkl','wb'))

    net = {'net_name':Arch,'loss': val_fn_loss, 'acc': val_fn_acc, 'logit':logits_fn, 'predict': predic_fun, 'gradLoss': fun_grad_loss, 'gradClassifier':fun_grad_classifir}
    data = {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test, 'meanPixel': meanpixel, 'clp_min':clip_min, 'clp_max':clip_max}
    return net, data

class FGS_family (object):
    """"
    Attributes:
        net_info contains some required functions like logits, cross-entropy loss, and etc,for generating adversaries
        data contains training and test sets
        dataset_type the name of dataset
        Arch the underlying network's architecture
        epsilon the hyperparameter,
        indx index of correctly classified samples
    """""
    def __init__(self, net_info, data, dataset_type,Arch , epsilon, indx ):
        self.net_info = net_info
        self.data = data
        self.dataset_type = dataset_type
        self.Arch = Arch
        self.epsilon = epsilon
        self.indx = indx

    def Generate_Attack (self,targeted=False):

        X = self.data['X_train'][self.indx]
        Y = self.data['y_train'][self.indx]
        meanpixel = self.data['meanPixel']
        clp_max = self.data['clp_max']
        clp_min = self.data['clp_min']
        epsilon = self.epsilon
        dataset_type = self.dataset_type
        indx = self.indx
        print(np.unique(Y))
        dustbin = len(np.unique(Y))
        fun_grad = self.net_info['gradLoss']
        p_softmax = self.net_info['predict']
        output_x = []
        output_y = []
        distortion = []
        outputVals = []
        Indx = []
        not_found = 0
        print('clp_min' + str(clp_min))
        print('clp_max' + str(clp_max))
        Avg_iter = 0
        total_iteration = 3
        for idx_init_img in range(len(X)):

            itc = 1
            pert = 0
            prtb_x, prtb_y = X[idx_init_img:idx_init_img + 1].copy(), Y[idx_init_img:idx_init_img + 1].copy()
            orig_x = X[idx_init_img:idx_init_img + 1].copy()

            target_y = np.argmin(p_softmax(orig_x)[0][:, :dustbin], axis=1).astype('uint8')
            print('= =========================')

            while itc <= total_iteration:
                num_classes = p_softmax(prtb_x)[0].shape[1]
                if targeted:
                    eta = epsilon * np.sign(fun_grad(prtb_x, target_y))
                else:
                    eta = epsilon * np.sign(fun_grad(prtb_x, prtb_y))

                pert = pert + eta

                if (itc >= 1):
                    if dataset_type == 'mnist':
                        alpha = 0.3  # MNIST
                    elif dataset_type == 'cifar10':
                        alpha = 0.05  # cifar10
                    pert = np.clip(pert, -alpha, alpha)

                if targeted:
                    prtb_x = orig_x - pert
                else:
                    prtb_x = orig_x + pert
                fooled_y = np.uint8(np.argmax(p_softmax(np.clip(prtb_x, clp_min, clp_max))[0], axis=1))

                if (prtb_y != fooled_y) and fooled_y != dustbin:

                    print('final fooled Target' + str(fooled_y))
                    print('True label' + str(prtb_y))
                    print('iteration : ' + str(itc))
                    prtb_x = np.clip(prtb_x, clp_min, clp_max)
                    Ec_dist = np.sqrt(np.mean((prtb_x[0] - orig_x[0]) ** 2))
                    print('Magnitude of distortion ' + str(Ec_dist))
                    distortion.append(Ec_dist)
                    outputVals.append(p_softmax(prtb_x)[0])
                    # print('Probability of adversarial'+str(np.max(p_softmax(prtb_x)[0], axis=1)))
                    output_x.append(prtb_x[0])
                    output_y.append(prtb_y)
                    Indx.append(indx[idx_init_img])
                    estimated_prediction = np.vstack(np.asarray(outputVals, dtype='float32'))
                    if np.mod(idx_init_img, 20) == 0:
                        print('***********img ' + str(idx_init_img) + '************')
                        print("not found =" + str(not_found))
                        print("Avg distortion {:.4f}, Avg confidence {:.4f} ".format(np.mean(distortion), np.mean(
                            np.max(estimated_prediction, axis=1), axis=0)))
                        print('***********************')
                    Avg_iter += itc

                    break
                itc += 1
            if itc > total_iteration:
                not_found += 1
                print(str(idx_init_img) + ' Not found****')

        print('=============== Average of iterations : ' + str(Avg_iter / len(outputVals)))
        data_fooled = zip(output_x, outputVals, output_y, distortion, Indx)
        # np.save(dataset_type+'_AdvTest_' + str(len(X))+'_espi_'+str(epsilon)+'_FastSign', output_x)

        pickle.dump({'data': data_fooled, 'meanpixel': meanpixel, 'Avg_iter': Avg_iter / len(outputVals)}, open(
            os.path.join(dataset_type + '_' + str(num_classes) + '_AdvSamples_' + str(len(X)) + '_espi_'+str(epsilon)+'_itc_'+str(total_iteration)+self.Arch+'_Targeted='+str(targeted)
                         +'.pkl'),'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        val_fn_acc = self.net_info['acc']
        all_acc = []
        for batch in util.iterate_minibatches(np.asarray(output_x), np.asarray(output_y)[0].flatten(), 100,
                                              shuffle=False):
            inputs, targets, indices = batch
            all_acc.append(val_fn_acc(inputs, targets))
        all_acc = np.asarray(np.hstack(all_acc))
        print('accuracy on adversaries  ' + str(np.mean(all_acc[0])))


def objective(r, *args):
    net = args[0]
    base = args[1]
    orig_label = args[2]
    c = args[3]
    r = r.reshape(base.shape)
    r = np.asanyarray(r)
    test_loss = net['loss']
    grad_input = net['gradLoss']
    # Frobenius norm is used [sum (a_ij)^2]^(1/2)
    obj = c * np.sqrt(np.sum(r ** 2)) + test_loss(base + r, orig_label)

    grad = grad_input(base + r, orig_label) + c * (r) / np.sqrt(np.sum(r ** 2))

    return obj, grad.flatten()


def LBFGS(net, data,  object_fun=objective, dataset_type=None,Arch=None, selected_sample =None, C=1):
    X = data['X_train'][selected_sample]
    y = data['y_train'][selected_sample]
    Mean_pixel = data['meanPixel']


    print('Generating adversarial examples by LBFGS')
    factr = 10.0
    pgtol = 1e-05
    if dataset_type =='cifar100':
        num_class = 100
    else:
        num_class = 10

    distortion = []
    output_x = []
    outputVals = []
    output_y = []
    indx = []
    c = []
    p_softmax = net['predict']

    for i in range(X.shape[0]):

        print('sample {}'.format(i))
        base = X[i:i + 1].copy()
        orig_label = y[i:i + 1].copy()
        # Initialization of distortion
        initial = (np.ones(base.shape, dtype='float32') * 1e-20).astype('float32')
        # Since -mean <base< 1-mean ==> -mean <base+r< 1-mean. So lower bound and upper bound for distortion are as follows:
        # lwr_bnd = -base - Mean_pixel
        # upr_bnd =  - base +1-Mean_pixel
        # bound = zip(lwr_bnd.flatten(), upr_bnd.flatten())
        lwr_bnd = -np.max(Mean_pixel)*np.ones(Mean_pixel.shape)- base
        upr_bnd = 1-np.min(Mean_pixel)*np.ones(Mean_pixel.shape) -base
        bound = zip(lwr_bnd.flatten(), upr_bnd.flatten())


        # select a label != orig_label
        # while True:
        #     fool_target = np.uint8(np.random.choice(range(num_class), 1))
        #     if fool_target != orig_label:
        #         print('Selected target {}, true target {}'.format(fool_target,orig_label))
        #         break


        fool_target = np.uint8(np.argmin(p_softmax(base)[0]))*np.ones(1).astype(int)

        print('target fool label ' + str(fool_target))
       # C = 2
        x, f, d = cnstOpt(object_fun, x0=initial.flatten().astype('float32'),
                          args=(net, base, fool_target, C),
                          bounds=bound, maxiter=10000, iprint=0, factr=factr,
                          pgtol=pgtol)
        print('prediction fool label {:.3f}'.format(p_softmax(x.reshape(base.shape) + base)[0][0,fool_target[0]]))
        print('prediction True label {:.3f}'.format(p_softmax(x.reshape(base.shape) + base)[0][0,orig_label[0]]))
        Ec_dist = np.sqrt(np.mean((x) ** 2))
        c.append(C)
        outputVals.append(p_softmax(x.reshape(base.shape) + base)[0])
        distortion.append(Ec_dist)
        print("Magnitude of distortion {:.6f}\n".format(Ec_dist))
        output_x.append(x.reshape(base.shape) + base+Mean_pixel)
        output_y.append(orig_label)
        indx.append(selected_sample[i])
        estimated_prediction = np.vstack(np.asarray(outputVals, dtype='float32'))
        print("Avg distortion {:.3f} AVG confidence {:.6f} \n".format(np.mean(distortion),
                                                                          np.mean(np.max(estimated_prediction,axis=1), axis=0)))
        print(np.max(p_softmax(x.reshape(base.shape) + base)[0]))
    data_fooled = zip(output_x, outputVals, output_y, distortion, indx, c)

    pickle.dump({'data':data_fooled,'meanpixel':Mean_pixel}, open(dataset_type +Arch+ '_'+str(X.shape[0])+'_LBFGS_LeastLikely_'+str(C)+'.pkl', 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

def DeepFool (net_info, data,  dataset_type=None,Arch=None, indx = None, hyp = 0.02 ):
    X = data['X_train'][indx]
    Y = data['y_train'][indx]
    clp_max = data['clp_max']
    clp_min = data['clp_min']

    all_classes = np.unique(Y)
    print(np.unique(Y))
    print('Generating adversarial examples by Deep Fool')
    fun_grad = net_info['gradClassifier']
    p_softmax = net_info['predict']
    logits = net_info['logit']
    output_x = []
    output_y = []
    distortion = []
    outputVals = []
    Indx = []
    dustbin = len(np.unique(Y))
    not_found = 0
    # all_classes = np.arange(0,dustbin+1)
    to_dustbin = 0
    for idx_init_img in range(len(X)):

        itc = 0
        accum_prtb = 0
        min_distance = np.inf
        prtb_x, orig_labl = X[idx_init_img:idx_init_img + 1].copy(), Y[idx_init_img:idx_init_img + 1].copy()
        orig_x = X[idx_init_img:idx_init_img + 1].copy()
        Labels = np.delete(all_classes, orig_labl)

        while itc < 50:
            itc += 1
            gradient_logit = fun_grad(prtb_x)

            for label in Labels:
                diff_grad = gradient_logit[0][label] - gradient_logit[0][orig_labl[0]]
                L2norm = np.sqrt(np.sum(diff_grad**2))

                # Dis_orig2fool = np.abs(p_softmax(prtb_x)[0][0,label] - p_softmax(prtb_x)[0][0,orig_labl[0]])/L2norm
                Dis_orig2fool = np.abs(logits(prtb_x)[0][0,label] - logits(prtb_x)[0][0,orig_labl[0]])/L2norm


                if Dis_orig2fool < min_distance:
                    min_distance = Dis_orig2fool
                    wk = diff_grad
                    Opt_label = label

            prtb =  (1e-4+min_distance)* (wk)/ np.sqrt(np.sum(wk.flatten()**2))
            accum_prtb += prtb

            prtb_x =  orig_x+ (1+0.02)*accum_prtb

            fooled_y = np.uint8(np.argmax(p_softmax(prtb_x)[0], axis=1))
            num_classes = p_softmax(prtb_x)[0].shape[1]


            if orig_labl[0] != fooled_y[0] and  fooled_y[0] != dustbin:
                # print('step size '+ str(W_L2norm[selected_indx]))
                if hyp !=0:
                    Final_prtb_x = orig_x+(1+hyp)*accum_prtb
                else:
                    Final_prtb_x = orig_x+(1+0.02)*accum_prtb

                Final_prtb_x = np.clip(Final_prtb_x, clp_min, clp_max)

                Ec_dist =np.sqrt(np.mean((Final_prtb_x[0] - orig_x[0]) ** 2))
                fooled_y = np.uint8(np.argmax(p_softmax(Final_prtb_x)[0], axis=1))

                print('fooled label {}'.format(fooled_y[0]))
                print('True label {}'.format(orig_labl[0]))

                print('Found at iteration ' + str(itc))
                if ~np.isnan(Ec_dist):
                    outputVals.append(p_softmax(Final_prtb_x)[0])
                    distortion.append(Ec_dist)
                    output_x.append(Final_prtb_x[0])
                    output_y.append(orig_labl)
                    Indx.append(indx[idx_init_img])
                estimated_prediction = np.vstack(np.asarray(outputVals, dtype='float32'))
                print('========= img '+str(idx_init_img)+'=======')
                print("Avg distortion (sezgdey) {:.4f}, Avg confidence {:.4f} ".format(np.mean(distortion), np.mean(np.max(estimated_prediction, axis=1),axis=0)))
                print('# of not found '+str(not_found))
                # print ('# of dustin fooled label'+str(to_dustbin))
                print('===============================')


                break
        if itc >= 50:
            not_found +=1
            print("******* Not found********")
            print ('fooled_y '+str(fooled_y))
            print ('optimum label '+str(Opt_label))
            print('step size' + str(min_distance))

    data_fooled = zip(output_x, outputVals, output_y, distortion, Indx)
    pickle.dump({'data':data_fooled,'meanpixel':data['meanPixel']}, open(os.path.join(dataset_type+'_'+str(num_classes)+Arch+  str(len(X))+'_overshoot'+str(hyp)+'_DeepFool_Test.pkl'),'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return output_x, output_y, outputVals, distortion







def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hm:d:a:n:p:", ["help", "method=","dataset=","arch=","PretrainedNet=", "parameter="])
    except getopt.GetoptError:

        raise ("Unkonwn arguments: see help => python GenAdv.py --help")


    for opt, arg in opts:

        if opt in ("-h, --help"):
            print(
                '========== Example: genAdv.py -m LBFGS -d cifar10 -a NiN -n Cifar_pretrained_CudaConvVersion.pkl -p 1 ================'
                + '\n -m or --method  indicates an attack for generating adversarial examples! optional methods [LBFGS, FastSign, DeepFool and Targeted FGS]'
                + '\n -d, --dataset takes a dataset. The code supports MNIST and cifar-10, cifar100, SVHN!'
                + '\n -a or --arch indicates architecture of the net'
                + '\n-n, --PretrainedNet takes a pkl filename, contains pre-trained weights of a net'
                + '\n -p or --parameter is the hyper parameter associated with the chosen attack ')
            sys.exit(2)

        elif opt in ("-m, --method"):
            print('Choosen method is '+str(arg))
            gener_method = arg
        elif opt in  ("-d, --dataset"):
            print('dataset is '+str(arg))
            dataset= arg
        elif opt in ("-n, --PretrainedNet"):
            print('pretrained net filename is '+str(arg))
            Pretrained_net = arg
        elif opt in ("-p, --parameter"):
            print('selected hyper-parameter is '+str(arg))
            hyperparameter = float(arg)

        elif opt in ("-a, --architecture"):
            Arch = str(arg)
            print('Selected Arch is '+str(Arch))



    # Load data and the pre-trained network
    net, data = Load_weights(Pretrained_net, Arch=Arch, dataset_type=dataset, BN=False)


    # Indicating the index of correctly classfied samples
    # indx_tr = pickle.load(open(data + '_Index_correcltyClassifiedTrain_' + Arch + '.pkl', 'rb'))
    indx = pickle.load(open(dataset+'_Index_correcltyClassifiedTest_' + Arch + '.pkl','rb'))

    if gener_method =='LBFGS':
        LBFGS(net, data,  object_fun=objective, dataset_type=dataset,Arch=Arch, selected_sample=indx[0], C = hyperparameter)

    elif gener_method == 'FastSign':

        FGS = FGS_family(net, data, dataset, Arch, hyperparameter, indx[0])
        FGS.Generate_Attack(targeted=False)

    elif gener_method == 'DeepFool':
        DeepFool(net,data, dataset,Arch, indx = indx[0],  hyp = hyperparameter)

    elif gener_method =='Targeted_FGS':
        FGS = FGS_family(net, data, dataset, Arch, hyperparameter, indx[0])

        FGS.Generate_Attack(targeted=True)
    else:
        raise Exception ('The requested algorithm has not been implemented yet!')


if __name__ == '__main__':
    main(sys.argv[1:])

