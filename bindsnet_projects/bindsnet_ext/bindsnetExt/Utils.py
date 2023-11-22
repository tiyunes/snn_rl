import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import OneHotEncoder
import h5py
import torch
import os
from bindsnet.encoding import poisson
from collections import OrderedDict

def round_decimals(value):
    i = 0
    while value < 1:
        value *= 10
        i += 1
    return i

def generate_search_space(init_pars:list, rn:float = 0.25, steps:int = 10)->OrderedDict:
    search_space_list = []
    for key in init_pars:
        bound1 = init_pars[key] * (1+rn)
        bound2 = init_pars[key] * (1-rn)
        
        lower_bound = min([bound1, bound2])
        upper_bound = max([bound1, bound2])
        
        search_space_list.append((key, hp.choice(key, np.linspace(lower_bound, upper_bound, steps))))
    return(OrderedDict(search_space_list))

class brbmGenerator(object):
    
    '''
    Spike encoder based on BernoulliRBM that is used to extract probabilities P(h|v) from the hidden layer.
    Those probabilities act as spike probabilities for each timestep dt.
    
    Input data should be scaled to fit in range(0,1).
    '''
    
    def __init__(self, 
                 n_components=256, 
                 learning_rate=0.1, 
                 batch_size=10, 
                 n_iter=10, 
                 verbose=0, 
                 random_state=None, 
                 T = 100, 
                 dt = 1,
                 mode = 'binary'):
    
        self.rbm = BernoulliRBM(n_components=n_components, 
                 learning_rate=learning_rate, 
                 batch_size=batch_size, 
                 n_iter=n_iter, 
                 verbose = verbose, 
                 random_state = random_state)
        
        self.T = T
        self.dt = dt
        self.mode = mode
        self.random_state = random_state
        
    def fit(self, x : np.ndarray):
        self.rbm.fit(x)
        
    def generate(self, x : np.ndarray)->np.ndarray:

        prob = self.rbm.transform(x)
        
        spike_array = []

        for step in range(int(self.T/self.dt)):
            if self.mode == 'binary':
                spikes = np.random.binomial(1, prob)
            elif self.mode == 'gaussian':
                spikes = (prob > np.random.normal(loc = 0., scale = 1., size = prob.shape)).astype(np.int)
            elif self.mode == 'uniform':
                spikes = (prob > np.random.uniform(low = 0., high = 1., size = prob.shape)).astype(np.int)
            elif self.mode == 'poisson':
                spikes = (prob > np.random.poisson(lam = 1., size = prob.shape)).astype(np.int)
            elif self.mode == 'innate':
                rng = np.random.RandomState(self.random_state)
                spikes = (rng.random_sample(size=prob.shape) < prob).astype(np.int)
            spike_array.append(np.expand_dims(spikes, axis = 1))
            
        return np.hstack(spike_array)

def loadMorphDataset(path, fold, pad = True, expand = True):
    
    ds = {}

    paddingLengths = []

    for tp in ['/tr.h5', '/vl.h5', '/ts.h5']:
        datapath = fold + tp
        f = h5py.File(os.path.join(path,datapath), 'r')

        textLengths = np.asarray([np.sqrt(x.shape[0]) for x in f['x2_flatten']]).astype(np.int)
        paddingLengths.append(textLengths.max())

        texts = []

        for i in range(len(f['x1_flatten'])):
            vecDim = int(f['x1_flatten'][i].shape[0] / textLengths[i])
            text = f['x1_flatten'][i].reshape((textLengths[i],vecDim))
            texts.append(text)
            del text

        y = np.argmax(np.asarray(f['y']), axis=1)

        ds[tp[1:3]] = (texts, y)

    if pad:

        maxLen = int(np.asarray(paddingLengths).max())

        for key in ds:
            newX = []

            for text in ds[key][0]:
                if not expand:
                    vecDim = text.shape[1]
                    newText = text
                    while len(newText) < maxLen:
                        newText = np.vstack((newText, np.zeros((1,vecDim))))
                    newX.append(newText)
                    del newText
                else:
                    vectDim = text.shape[1]*2
                    newText = []
                    for word in text:
                        newWord = []
                        for x_i in word:
                            if x_i == 1:
                                newWord.append(1)
                                newWord.append(0)
                            else:
                                newWord.append(0)
                                newWord.append(1)
                        newText.append(newWord)
                    newText = np.asarray(newText)
                    addition = list(np.ones(vectDim))
                    newAddition = np.asarray([addition[i]*(i%2) for i in range(vectDim)]).reshape((1,vectDim))
                    while len(newText) < maxLen:
                        newText = np.vstack((newText, newAddition))
                    newX.append(newText)
                    del newText
                    del addition
                    del newAddition
                    
                    
            oldY = ds[key][1]
            ds[key] = (np.asarray(newX).astype(np.int), oldY)
            del oldY
            del newX
            

    return ds
    
    
def getSpikes(sp, n_classes, y, dt = 0.1, T = 100.):

    spike_times = [[int(sp['input'][i][j]/dt) for j in range(0, len(sp['input'][i]))] for i in range(0, len(sp['input']))]

    #generate spike patterns

    spikes = np.zeros((len(spike_times), int(T/dt), len(spike_times[0])))

    for i in range(len(spikes)):
        for j in range(len(spikes[i])):
            for k in range(len(spikes[i][j])):
                if spike_times[i][k] == j:
                    spikes[i][j][k] = 1

    #generate teacher spikes

    teacher_spikes = np.zeros((len(spike_times), int(T/dt), n_classes))


    minimal_times = np.min(spike_times, axis = 1)

    for i in range(len(teacher_spikes)):
        for j in range(len(teacher_spikes[i])):
            if minimal_times[i] == j:
                #print(y[i])
                teacher_spikes[i][j][y[i]] = 1

    #convert outputs into tensors
    X = torch.as_tensor(spikes, dtype = torch.uint8)
    Y = torch.as_tensor(teacher_spikes, dtype = torch.uint8)
    
    return (X, Y)

            
def getClasses(S_init):
    result = []
    try:
        S = S_init
        for i in range(len(S)):
            m, k = torch.where(S[i] == S[i].max())
            cl = k[torch.where(m == m.min())]
            if len(cl) > 1:
                cl = torch.as_tensor([99])
            result.append(cl)
    except ValueError:
        S = S_init.unsqueeze(-1)
        for i in range(len(S)):
            m, k = torch.where(S[i] == S[i].max())
            cl = k[torch.where(m == m.min())]
            if len(cl) > 1:
                cl = torch.as_tensor([99])
            result.append(cl)
    return torch.vstack(result)

class ReceptiveFieldsConverter(object):
    """
        Class for receptive fields data conversion
        Author: dartl0l
    """
    def __init__(self, sigma2, max_x, n_fields, k_round, scale=1.0,
                 reshape=True, reverse=False, no_last=False):
        self.sigma2 = sigma2
        self.max_x = max_x
        self.n_fields = n_fields
        self.k_round = k_round
        self.scale = scale
        self.reshape = reshape
        self.reverse = reverse
        self.no_last = no_last

    def get_gaussian(self, x, sigma2, mu):
        return (1 / np.sqrt(2 * sigma2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma2))

    def convert(self, x, y):
        """
            Function must be updated to O(n) complexity 
        """

        h_mu = self.max_x / (self.n_fields - 1)

        max_y = np.round(self.get_gaussian(h_mu, self.sigma2, h_mu), 0)

        mu = np.tile(np.linspace(0, self.max_x, self.n_fields), len(x[0]))
        x = np.repeat(x, self.n_fields, axis=1)

        assert len(mu) == len(x[0])

        if self.reverse:
            x = np.round(self.get_gaussian(x, self.sigma2, mu), self.k_round)
            if self.no_last:
                mask = x < 0.1
                x[mask] = np.nan
        else:
            x = max_y - np.round(self.get_gaussian(x, self.sigma2, mu), self.k_round)
            if self.no_last:
                mask = x > max_y - 0.09
                x[mask] = np.nan

        x *= self.scale
        if self.reshape:
            output = {'input': x.reshape(x.shape[0], x.shape[1], 1),
                      'class': y}
        else:
            output = {'input': x,
                      'class': y}
        return output  # , max_y
    
    
def loadMorphDataset(path, fold, pad = True, extend_size = False, crop = False):
    
    ds = {}
    
    paddingLengths = []
    
    for tp in ['/tr.h5', '/vl.h5', '/ts.h5']:
        datapath = fold + tp
        f = h5py.File(os.path.join(path,datapath), 'r')
        
        textLengths = np.asarray([np.sqrt(x.shape[0]) for x in f['x2_flatten']]).astype(np.int)
        paddingLengths.append(textLengths.max())
        
        texts = []
        
        for i in range(len(f['x1_flatten'])):
            vecDim = int(f['x1_flatten'][i].shape[0] / textLengths[i])
            text = f['x1_flatten'][i].reshape((textLengths[i],vecDim))
            texts.append(text)
            del text
            
        y = np.argmax(np.asarray(f['y']), axis=1)
        
        ds[tp[1:3]] = (texts, y)
        
    if pad:
        
        maxLen = int(np.asarray(paddingLengths).max())
        
        for key in ds:
            newX = []
            
            for text in ds[key][0]:
                if extend_size:
                    vecDim = maxLen
                    addDim = vecDim - text.shape[1]
                    sub = np.zeros(addDim)
                    newText = np.asarray([np.concatenate((word, sub)) for word in text])
                else:
                    vecDim = text.shape[1]
                    newText = text
                while len(newText) < maxLen:
                    newText = np.vstack((newText, np.zeros((1,vecDim))))
                newX.append(newText)
                del newText
            oldY = ds[key][1]
            ds[key] = (np.asarray(newX), oldY)
            del oldY
            del newX
            
    if crop:
        for key in ds:
            vecDim = ds[key][0].shape[2]
            ds[key] = (ds[key][0][:,:vecDim, :vecDim], ds[key][1])
            
    return ds

def encodeTexts(corpus, time=10, dt=0.1, intensity = 20):
    new_texts = []
    for text in corpus:
        x = torch.as_tensor(text * intensity).long()
        enc_x = poisson(datum = x, time = time, dt = dt)
        new_texts.append(enc_x.unsqueeze(0))
        
    return torch.cat(new_texts, axis = 0)