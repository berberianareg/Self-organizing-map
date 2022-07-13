"""Self-organizing map for unsupervised machine learning.

Notes
-----
    This script is version v0. It provides the base for all subsequent
    iterations of the project.

Requirements
------------
    See "requirements.txt"

Websites
--------
    https://en.wikipedia.org/wiki/Self-organizing_map

Relevant papers
--------
    https://doi.org/10.1016/j.neunet.2009.06.011

"""
#%% import libraries and modules
import numpy as np
import matplotlib.pyplot as plt

#%% build the SOM model
class SOM:
    """Self-organizing map class."""
    
    def __init__(self):
        """Specify default parameters."""
        # network parameters
        self.dimX = 36                                                          # dimension of input layer X (default: 36)
        self.dimY = 100                                                         # dimension of output layer Y (default: 100)
        self.wMin = 0.05                                                        # min connection weight (default: 0.05)
        self.wMax = 0.2                                                         # max connection weight (default: 0.2)
        # learning parameters
        self.nbIter = 100                                                       # nb of learning iterations (default: 100)
        self.etaTau = 15                                                        # time constant controlling learning rate decrease across trials (default: 15)
        self.eta0 = 0.5                                                         # initial learning rate (default: 0.5)
        self.eta = self.eta0 * np.exp(-np.arange(self.nbIter)/self.etaTau)      # learning rate across trials
        # neighbourhood function parameters
        self.sigmaTau = 10                                                      # time constant controlling neighbourhood size decrease across trials (default: 10)
        self.sigma0 = 10                                                        # initial topological neighbourhood size (default: 10)
        self.sigma = self.sigma0 * np.exp(-np.arange(self.nbIter)/self.sigmaTau)# topological neighbourhood size across trials
        # dimension of prototypes and examplars 
        self.dimProto = 4                                                       # dimension of prototypes (default: 4)
        self.dimExamp = 5                                                       # dimension of examplars (default: 5)
        # dimension of min and max "pixel flips"
        self.dimMinFlip = 1                                                     # dimension of min "pixel flips" (default: 1)
        self.dimMaxFlip = 4                                                     # dimension of max "pixel flips" (default: 4)
        # random uniform distribution parameters
        self.ruMin = 0                                                          # random uniform distribution min (default: 0)
        self.ruMax = 10                                                         # random uniform distribution max (default: 10)
        self.ruLim = 5                                                          # random uniform distribution limit (default: 5)

    def inputs(self):
        """Construct artificial input patterns for prototype-based categorization."""
        # dimension of prototypes and examplars 
        dimProto = self.dimProto
        dimExamp = self.dimExamp
        # random uniform distribution parameters
        ruMin = self.ruMin
        ruMax = self.ruMax
        ruLim = self.ruLim
        # dimension of min and max "pixel flips"
        dimMinFlip = self.dimMinFlip
        dimMaxFlip = self.dimMaxFlip
        # input dimension
        dimX = self.dimX
        # generate examplars from prototypes
        inp = [[] for iProto in range(dimProto)]                                # empty list for storing input vectors
        for iProto in range(dimProto):
            np.random.seed(iProto)                                              # random seed
            inp[iProto] = np.tile((np.random.uniform(ruMin,ruMax,size=dimX) > ruLim).astype(int),(dimExamp,1)) # generate binary input vectors from uniform distribution 
            for iExamp in range(1,dimExamp):
                np.random.seed(iExamp)                                          # random seed 
                dimFlip = np.random.randint(dimMinFlip,dimMaxFlip+1,1)          # dimension of "pixel flips"
                idxFlip = np.random.choice(dimX,dimFlip,replace=False)          # from dimX pixels, specify index/indices of dimFlip pixels to "flip" 
                inp[iProto][iExamp][idxFlip] = 1-(inp[iProto][0][idxFlip])      # perform dimFlip "pixel flips" on dimExamp-1 examplars 
        return inp
 
    def learning(self, inp):
        """Learning procedure."""
        # network parameters        
        dimX = self.dimX
        dimY = self.dimY
        wMin = self.wMin
        wMax = self.wMax
        np.random.seed(1)                                                       # random seed
        w = np.random.uniform(low=wMin,high=wMax,size=(dimX,dimY))              # initial random uniform connection weight matrix
        # dimension of prototypes and examplars 
        dimProto = self.dimProto
        dimExamp = self.dimExamp
        # learning parameters
        nbIter = self.nbIter
        eta = self.eta
        # neighbourhood function parameters
        sigma = self.sigma
        np.random.seed(16)                                                      # random seed
        randSample = np.random.choice(dimExamp,dimExamp,replace=False)          # random selection of examplars 
        iExamp = 0
        while iExamp < dimExamp:
            iProto = 0
            while iProto < dimProto:
                iIter = 0
                while iIter < nbIter:
                    # find the position of the winning output unit whos weight vector matches the input vector most closely (norm of their differences)
                    idxWinner = np.argmin((np.square(inp[iProto][randSample[iExamp]] - w.T)).sum(axis=1))
                    # topological neighbourhood function (Gaussian centering around winning node, and decreasing in all directions from it)
                    h = np.transpose([np.exp(-np.square(np.arange(dimY) - idxWinner)/(2*np.square(sigma[iIter])))])
                    # update connection weights of winning unit and its proximal neighbours
                    w = w + eta[iIter] * (h * (inp[iProto][randSample[iExamp]] - w.T)).T
                    iIter += 1
                iProto += 1
            iExamp += 1
        return w

    def outputs(self, inp, w):
        """Compute activations and network outputs."""
        # dimension of prototypes and examplars
        dimProto = self.dimProto
        dimExamp = self.dimExamp
        # network parameters
        dimY = self.dimY
        # compute activation
        activation = np.dot(np.vstack(np.array(inp)),w)                                 
        # for each output unit, find the category containing the examplar that yields the highest level of activation
        data = np.argmax(activation,axis=0)
        # categorize data into n different bins of equal size
        out = np.digitize(x=data, bins=np.linspace(0,dimProto*dimExamp,dimExamp)).reshape(int(np.sqrt(dimY)), int(np.sqrt(dimY)))
        return out

#%% run the SOM model
som = SOM()                                                                     # instantiate SOM class
inp = som.inputs()                                                              # input patterns
w = som.learning(inp)                                                           # perform learning
out = som.outputs(inp, w)                                                       # output patterns

#%% plot figures
fig, ax = plt.subplots(nrows=som.dimProto,ncols=som.dimExamp,sharex=True,sharey=True)
for iProto in range(som.dimProto):
    for iExamp in range(som.dimExamp):
        ax[iProto,iExamp].imshow(np.reshape(inp[iProto][iExamp],(int(np.sqrt(som.dimX)),int(np.sqrt(som.dimX)))))
        ax[iProto,0].set_ylabel('Cat. {}'.format(iProto+1))        
plt.xticks([]),plt.yticks([])
fig.suptitle('Input patterns')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
plt.imshow(out)
plt.title('2D topology')
plt.xticks([])
plt.yticks([])
plt.show()

del ax, fig, iExamp, iProto


