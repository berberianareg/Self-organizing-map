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
    
    def __init__(self,
                 input_size=36, output_size=100,
                 min_weight=0.05, max_weight=0.2,
                 num_iterations=100, learning_rate_decay=15, initial_learning_rate=0.5,
                 neighborhood_decay=10, initial_neighborhood_size=10,
                 num_prototypes=4, num_examples=5,
                 min_num_pixel_flips=1, max_num_pixel_flips=4,
                 min_random_uniform=0, max_random_uniform=10, random_uniform_limit=5):
        """Specify default parameters."""
        # input and output parameters
        self.input_size = input_size                                            # dimension of input layer X (default: 36)
        self.output_size = output_size                                          # dimension of output layer Y (default: 100)
        # connection weights parameters
        self.min_weight = min_weight                                            # min connection weight (default: 0.05)
        self.max_weight = max_weight                                            # max connection weight (default: 0.2)
        # learning parameters
        self.num_iterations = num_iterations                                    # nb of learning iterations (default: 100)
        self.learning_rate_decay = learning_rate_decay                          # time constant controlling learning rate decrease over trials (default: 15)
        self.initial_learning_rate = initial_learning_rate                      # initial learning rate (default: 0.5)
        self.learning_rates = self.initial_learning_rate * np.exp(-np.arange(self.num_iterations)/self.learning_rate_decay) # learning rate over trials
        # neighbourhood function parameters
        self.neighborhood_decay = neighborhood_decay                            # time constant controlling neighbourhood size decrease over trials (default: 10)
        self.initial_neighborhood_size = initial_neighborhood_size              # initial topological neighbourhood size (default: 10)
        self.sigma = self.initial_neighborhood_size * np.exp(-np.arange(self.num_iterations)/self.neighborhood_decay) # topological neighbourhood size over trials
        # dimension of prototypes and examplars
        self.num_prototypes = num_prototypes                                    # dimension of prototypes (default: 4)
        self.num_examples = num_examples                                        # dimension of examplars (default: 5)
        # dimension of min and max "pixel flips"
        self.min_num_pixel_flips = min_num_pixel_flips                          # dimension of min "pixel flips" (default: 1)
        self.max_num_pixel_flips = max_num_pixel_flips                          # dimension of max "pixel flips" (default: 4)
        # random uniform distribution parameters
        self.min_random_uniform = min_random_uniform                            # random uniform distribution min (default: 0)
        self.max_random_uniform = max_random_uniform                            # random uniform distribution max (default: 10)
        self.random_uniform_limit = random_uniform_limit                        # random uniform distribution limit (default: 5)

    def inputs(self):
        """Construct artificial input patterns for prototype-based categorization."""
        # array for storing input data
        input_data = np.zeros([self.num_prototypes, self.num_examples, self.input_size])
        for iProto in range(self.num_prototypes):
            # random seed
            np.random.seed(iProto)
            # generate uniform data
            uniform_data = np.random.uniform(low=self.min_random_uniform, high=self.max_random_uniform, size=self.input_size)
            # transform uniform data to binary data
            binary_data = (uniform_data > self.random_uniform_limit).astype(int)
            # create copies of the binary data
            input_data[iProto,:,:] = np.tile(binary_data, (self.num_examples, 1))
            for iExamp in range(1, self.num_examples):
                # random seed
                np.random.seed(iExamp)
                # specify number of pixels to flip
                dimFlip = np.random.randint(low=self.min_num_pixel_flips, high=self.max_num_pixel_flips+1, size=1)
                # specify index of pixels to flip
                idxFlip = np.random.choice(self.input_size, size=dimFlip, replace=False)
                # perform pixel flip (0 -> 1; 1 -> 0)
                input_data[iProto,iExamp,idxFlip] = 1-(input_data[iProto,0,idxFlip])
        return np.vstack(input_data)
    
    def train(self, input_data):
        """Learning procedure."""
        # initial random uniform connection weight matrix
        weights = np.random.uniform(low=self.min_weight, high=self.max_weight, size=(self.output_size, self.input_size))
        # random sequence of input selection
        random_sample = np.random.choice(self.num_examples * self.num_prototypes,
                                         self.num_examples * self.num_prototypes, replace=False)
        input_index = 0
        while input_index < self.num_examples * self.num_prototypes:
            iteration_index = 0
            while iteration_index < self.num_iterations:
                # find the position of the winning unit whos weight vector matches the input vector most closely (norm of their differences)
                winning_unit_index = np.argmin((np.square(input_data[random_sample[input_index]] - weights)).sum(axis=1))
                # specify the position of all output units
                unit_indices = np.arange(self.output_size)
                # compute distance between winning unit and all output units
                neighborhood_distance = unit_indices - winning_unit_index
                # topological neighbourhood function (Gaussian centering around winning unit, and decreasing in all directions from it)
                h = np.exp(-np.square(neighborhood_distance)/(2*np.square(self.sigma[iteration_index])))
                # update connection weights of winning unit and its proximal neighbours
                weights = weights + self.learning_rates[iteration_index] * (h[:,np.newaxis] * (input_data[random_sample[input_index]] - weights))
                iteration_index += 1
            input_index += 1
        return weights

    def test(self, input_data, weights):
        """Compute activations and network outputs."""
        # compute activation
        activation = np.dot(input_data, weights.T)
        # for each output unit, find the category containing the examplar that yields the highest level of activation
        data = np.argmax(activation, axis=0)
        # categorize data into n different bins of equal size
        out = np.digitize(x=data, bins=np.linspace(0, self.num_prototypes*self.num_examples, self.num_examples)).reshape(int(np.sqrt(self.output_size)), int(np.sqrt(self.output_size)))
        return out

#%% run the SOM model
som = SOM()                                                                     # instantiate SOM class
input_data = som.inputs()                                                       # input patterns
weights = som.train(input_data)                                                 # perform learning
output_data = som.test(input_data, weights)                                             # perform testing

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

fig = plt.figure()
for input_index in range(1, som.num_examples * som.num_prototypes + 1):
    fig.add_subplot(som.num_prototypes, som.num_examples, input_index)
    plt.imshow(input_data[input_index-1].reshape(int(np.sqrt(som.input_size)), int(np.sqrt(som.input_size))))
    plt.xticks(ticks=[]), plt.yticks(ticks=[])
fig.suptitle('Input patterns')
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'figure_1'))

fig = plt.figure()
plt.imshow(output_data)
plt.title('2D topology')
plt.xticks(ticks=[]), plt.yticks(ticks=[])
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'figure_2'))

del fig, input_index
