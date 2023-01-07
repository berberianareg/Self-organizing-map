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

Relevant paper
--------
    https://doi.org/10.1016/j.neunet.2009.06.011

"""
#%% import libraries and modules
import numpy as np
import matplotlib.pyplot as plt
import os

#%% build the SOM model
class SOM:
    """Self-organizing map class."""
    
    def __init__(self, dimX=36, dimY=100, wMin=0.05, wMax=0.2, nbIter=100,
                 etaTau=15, eta0=0.5, sigmaTau=10, sigma0=10, dimProto=4,
                 dimExamp=5, dimMinFlip=1, dimMaxFlip=4, ruMin=0, ruMax=10,
                 ruLim=5):
        """Specify default parameters."""
        # network parameters
        self.dimX = dimX                                                        # dimension of input layer X (default: 36)
        self.dimY = dimY                                                        # dimension of output layer Y (default: 100)
        self.wMin = wMin                                                        # min connection weight (default: 0.05)
        self.wMax = wMax                                                        # max connection weight (default: 0.2)
        # learning parameters
        self.nbIter = nbIter                                                    # nb of learning iterations (default: 100)
        self.etaTau = etaTau                                                    # time constant controlling learning rate decrease across trials (default: 15)
        self.eta0 = eta0                                                        # initial learning rate (default: 0.5)
        self.eta = self.eta0 * np.exp(-np.arange(self.nbIter)/self.etaTau)      # learning rate across trials
        # neighbourhood function parameters
        self.sigmaTau = sigmaTau                                                # time constant controlling neighbourhood size decrease across trials (default: 10)
        self.sigma0 = sigma0                                                    # initial topological neighbourhood size (default: 10)
        self.sigma = self.sigma0 * np.exp(-np.arange(self.nbIter)/self.sigmaTau)# topological neighbourhood size across trials
        # dimension of prototypes and examplars
        self.dimProto = dimProto                                                # dimension of prototypes (default: 4)
        self.dimExamp = dimExamp                                                # dimension of examplars (default: 5)
        # dimension of min and max "pixel flips"
        self.dimMinFlip = dimMinFlip                                            # dimension of min "pixel flips" (default: 1)
        self.dimMaxFlip = dimMaxFlip                                            # dimension of max "pixel flips" (default: 4)
        # random uniform distribution parameters
        self.ruMin = ruMin                                                      # random uniform distribution min (default: 0)
        self.ruMax = ruMax                                                      # random uniform distribution max (default: 10)
        self.ruLim = ruLim                                                      # random uniform distribution limit (default: 5)

    def inputs(self):
        """Construct artificial input patterns for prototype-based categorization."""
        # empty list for storing input data
        inp = [[] for iProto in range(self.dimProto)]
        for iProto in range(self.dimProto):
            # random seed
            np.random.seed(iProto)
            # generate uniform data
            uniform_data = np.random.uniform(low=self.ruMin, high=self.ruMax, size=self.dimX)
            # transform uniform data to binary data
            binary_data = (uniform_data > self.ruLim).astype(int)
            # create copies of the binary data
            inp[iProto] = np.tile(binary_data, (self.dimExamp, 1))
            for iExamp in range(1,self.dimExamp):
                # random seed
                np.random.seed(iExamp)
                # specify number of pixels to flip
                dimFlip = np.random.randint(low=self.dimMinFlip, high=self.dimMaxFlip+1, size=1)
                # specify index of pixels to flip
                idxFlip = np.random.choice(self.dimX, size=dimFlip, replace=False)
                # perform pixel flip (0 -> 1; 1 -> 0)
                inp[iProto][iExamp][idxFlip] = 1-(inp[iProto][0][idxFlip])
        return inp
 
    def train(self, inp):
        """Learning procedure."""
        # random seed
        np.random.seed(1)
        # initial random uniform connection weight matrix
        weights = np.random.uniform(low=self.wMin, high=self.wMax, size=(self.dimX, self.dimY))
        # random seed
        np.random.seed(16)
        # random sequence of examplar selection
        randSample = np.random.choice(self.dimExamp, self.dimExamp, replace=False)
        iExamp = 0
        while iExamp < self.dimExamp:
            iProto = 0
            while iProto < self.dimProto:
                iIter = 0
                while iIter < self.nbIter:
                    # find the position of the winning unit whos weight vector matches the input vector most closely (norm of their differences)
                    idxWinner = np.argmin((np.square(inp[iProto][randSample[iExamp]] - weights.T)).sum(axis=1))
                    # specify the position of all output units
                    idxUnits = np.arange(self.dimY)
                    # compute distance between winning unit and all output units
                    neighbourhood_distance = idxUnits - idxWinner
                    # topological neighbourhood function (Gaussian centering around winning unit, and decreasing in all directions from it)
                    h = np.exp(-np.square(neighbourhood_distance)/(2*np.square(self.sigma[iIter])))
                    # update connection weights of winning unit and its proximal neighbours
                    weights = weights + self.eta[iIter] * (np.transpose([h]) * (inp[iProto][randSample[iExamp]] - weights.T)).T
                    iIter += 1
                iProto += 1
            iExamp += 1
        return weights

    def test(self, inp, weights):
        """Compute activations and network outputs."""
        # compute activation
        activation = np.dot(np.vstack(np.array(inp)), weights)
        # for each output unit, find the category containing the examplar that yields the highest level of activation
        data = np.argmax(activation, axis=0)
        # categorize data into n different bins of equal size
        out = np.digitize(x=data, bins=np.linspace(0, self.dimProto*self.dimExamp, self.dimExamp)).reshape(int(np.sqrt(self.dimY)), int(np.sqrt(self.dimY)))
        return out

#%% run the SOM model
som = SOM()                                                                     # instantiate SOM class
inp = som.inputs()                                                              # input patterns
weights = som.train(inp)                                                        # perform learning
out = som.test(inp, weights)                                                    # perform testing

#%% plot figures

cwd = os.getcwd()                                                               # get current working directory
fileName = 'images'                                                             # specify filename

if os.path.exists(os.path.join(cwd, fileName)) == False:                        # if path does not exist
    os.makedirs(fileName)                                                       # create directory with specified filename
    os.chdir(os.path.join(cwd, fileName))                                       # change cwd to the given path
    cwd = os.getcwd()                                                           # get current working directory
else:
    os.chdir(os.path.join(cwd, fileName))                                       # change cwd to the given path
    cwd = os.getcwd()                                                           # get current working directory

fig, ax = plt.subplots(nrows=som.dimProto,ncols=som.dimExamp,sharex=True,sharey=True)
for iProto in range(som.dimProto):
    for iExamp in range(som.dimExamp):
        ax[iProto,iExamp].imshow(np.reshape(inp[iProto][iExamp],(int(np.sqrt(som.dimX)),int(np.sqrt(som.dimX)))))
        ax[iProto,0].set_ylabel('Cat. {}'.format(iProto+1))
plt.xticks([]),plt.yticks([])
fig.suptitle('Input patterns')
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'figure_1'))

fig, ax = plt.subplots()
plt.imshow(out)
plt.title('2D topology')
plt.xticks([])
plt.yticks([])
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'figure_2'))

del ax, fig, iExamp, iProto
