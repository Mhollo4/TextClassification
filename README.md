# TextClassification
# Objective 
implement an artificial neural network to automatically classify a set of non-functional requirements into its seperate categories as described below:
Access control
Maintainability
Legal
Look and feel
Reliability
Usability
Operational 
Recoverability 
Privacy 
Security
Other nonfunctional

# Implementation
to implement this idea I utilized python and Jupyter notebook to compile and run the program. I took a series of steps which are later explained to complete this implementation.
1. Download and refer to libraries we need
2. Provide training data
3. Organize The data data
4. Iterate: code, test results, tune the model
5. Abstract

# Download and refer to libraries
The most important libraries that I downloaded were:
Numpy for our algorithmic functions
Nltk for pre data pre-preprocessing 
Sklearn for our accuracy predictions and k fold splitting 

# Providing training data
The data set is a series of sentences of non functional requirements from business use cases provided by my professor. In order to provide clean Training data I took the following steps:
1. Setting all characters to lowercase
2. Removing any duplicates
3. Stemming (removing prefixes, suffixes from all words)
4. Removing default stop words (the, and, a)
5. Removing custom stop words (system, product, shall, etc.)
6. Making sure syntax is correct

# Orgnaizing Data
We organize our data  into a bag of words for each sentence which is then compared to each uniquely stemmed words. 

# Iterate Test Code
first I set up a sigmoid function and its derivative 

def f(x):
    return 1 / (1 + np.exp(-x))
def f_deriv(x):
    return f(x) * (1 - f(x))

    
Randomly initialise the weights for each layer W(l)
While iterations < iteration limit:
1. Set ΔW and Δb to zero
2. For samples 1 to m:
a. Perform a feed foward pass through all the nl layers. Store the activation function outputs h(l)
b. Calculate the δ(nl) value for the output layer
c. Use backpropagation to calculate the δ(l) values for layers 2 to nl−1
d. Update the ΔW(l) and Δb(l) for each layer
3. Perform a gradient descent step using:

W(l)=W(l)–α[1mΔW(l)]
b(l)=b(l)–α[1mΔb(l)]
So the first step is to initialise the weights for each layer. To make it easy to organise the various layers, we’ll use Python dictionary objects (initialised by {}). Finally, the weights have to be initialised with random values – this is to ensure that the neural network will converge correctly during training. We use the numpy library random_sample function to do this. The weight initialisation code is shown below:


import numpy.random as r
def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b


The next step is to set the mean accumulation values ΔW and Δb to zero (they need to be the same size as the weight and bias matrices):


def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b
    
If we now step into the gradient descent loop, the first step is to perform a feed forward pass through the network.


def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise, 
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)  
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l)) 
    return h, z


Finally, we have to then calculate the output layer delta δ(nl) and any hidden layer delta values δ(l) to perform the backpropagation pass:

def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out)

def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)


Now we can put all the steps together into the final function:

def train_nn(nn_structure, X, y, iter_num=1500, alpha=0.25):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, to be used in the
            # gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis])) 
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


The function above deserves a bit of explanation. First, we aren’t setting a termination of the gradient descent process based on some change or precision of the cost function. Rather, we are just running it for a set number of iterations (1,500 in this case) and we’ll monitor how the average cost function changes as we progress through the training (avg_cost_func list in the above code). In each iteration of the gradient descent, we cycle through each training sample (range(len(y)) and perform the feed forward pass and then the backpropagation. The backpropagation step is an iteration through the layers starting at the output layer and working backwards – range(len(nn_structure), 0, -1). We calculate the average cost, which we are tracking during the training, at the output layer (l == len(nn_structure)). We also update the mean accumulation values, ΔW and Δb, designated as tri_W and tri_b, for every layer apart from the output layer (there are no weights connecting the output layer to any further layer).

Finally, after we have looped through all the training samples, accumulating the tri_W and tri_b values, we perform a gradient descent step change in the weight and bias values:
W(l)=W(l)–α[1mΔW(l)]

b(l)=b(l)–α[1mΔb(l)]
After the process is completed, we return the trained weight and bias values, along with our tracked average cost for each iteration. Now it’s time to run the function – NOTE: this may take a few minutes depending on the capabilities of your computer.

W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)

Now we can have a look at how the average cost function decreased as we went through the gradient descent iterations of the training, slowly converging on a minimum in the function:

plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()

as a restult I have trained this neural network to be 91% accurate 
