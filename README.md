# Corona-Virus-Image-Detection

On March 11, four days before this article was written, the World Health Organization (W.H.O.) declared Coronavirus disease 2019 (NCOV-19) a pandemic characterized by the rapid and global spread of the novel coronavirus around the world. As governments scramble to close borders, implement contact tracing, and increase awareness of personal hygiene in an effort to contain the spread of the virus, the spread of the virus is still unfortunately expected to increase until a vaccine can be developed and deployed owing to different standards of implementing these policies for each country.

As actual daily cases are expected to increase throughout the world, one significant factor that limits diagnosis is the duration of pathology tests for the virus, which are carried out in laboratories usually in city centers that demand time-consuming precision. This causes significant problems, chiefly the fact that individuals who are carriers cannot be isolated earlier, and thus they are able to infect more people during that critical period of unrestricted movement. Another problem would be the costly large-scale implementation of the current diagnosis procedure. Arguably, the most vulnerable are people in remote areas within developing countries who generally have inferior healthcare and access to diagnosis. Having a single infection may be detrimental for these communities, and having access to diagnosis will at least give them a fighting chance against the virus.


# Convolution Layer
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

# Neural Network
A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes.[1] Thus a neural network is either a biological neural network, made up of real biological neurons, or an artificial neural network, for solving artificial intelligence (AI) problems. The connections of the biological neuron are modeled as weights. A positive weight reflects an excitatory connection, while negative values mean inhibitory connections. All inputs are modified by a weight and summed. This activity is referred as a linear combination. Finally, an activation function controls the amplitude of the output. For example, an acceptable range of output is usually between 0 and 1, or it could be −1 and 1.

![Image of a Convoluted Neural Network](https://adeshpande3.github.io/assets/Cover.png)


# Standardize
A standard approach is to scale the inputs to have mean 0 and a variance of 1. Also linear decorrelation/whitening/pca helps a lot.

# Dense Layer

A dense layer is just a regular layer of neurons in a neural network. Each neuron recieves input from all the neurons in the previous layer, thus densely connected. The layer has a weight matrix W, a bias vector b, and the activations of previous layer a. The following is te docstring of class Dense from the keras documentation:
output = activation(dot(input, kernel) + bias)where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer.

# Dropout Layer 

Dropout is a a technique used to tackle Overfitting . The Dropout method in keras.layers module takes in a float between 0 and 1, which is the fraction of the neurons to drop. Below is the docstring of the Dropout method from the documentation:
Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.

# Compile

Every Neural Network should be compiled before Training it on a Dataset. During Compilation wer have to provide our neural network with an optimizer, a loss function as well as the Metrics that we need to observe during Training

## Adam
Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.

## Binary Cross-Entropy
Also called Sigmoid Cross-Entropy loss. It is a Sigmoid activation plus a Cross-Entropy loss. Unlike Softmax loss it is independent for each vector component (class), meaning that the loss computed for every CNN output vector component is not affected by other component values. That’s why it is used for multi-label classification, were the insight of an element belonging to a certain class should not influence the decision for another class. It’s called Binary Cross-Entropy Loss because it sets up a binary classification problem between C′=2 classes for every class in C, as explained above. So when using this Loss, the formulation of Cross Entroypy Loss for binary problems is often used:

## Sigmoid
A sigmoid function is a type of activation function, and more specifically defined as a squashing function. Squashing functions limit the output to a range between 0 and 1, making these functions useful in the prediction of probabilities.
![Sigmoid](https://www.researchgate.net/profile/Tali_Leibovich-Raveh/publication/325868989/figure/fig2/AS:639475206074368@1529474178211/A-Basic-sigmoid-function-with-two-parameters-c1-and-c2-as-commonly-used-for-subitizing.png)

# CoronaVirus has been a disaster and we should all come together to make this world a better place.

![Accuracy](https://raw.githubusercontent.com/Mkaif-Agb/Corona-Virus-Image-Detection/master/acc.png)
![Loss](https://raw.githubusercontent.com/Mkaif-Agb/Corona-Virus-Image-Detection/master/loss.png)
![Health](https://raw.githubusercontent.com/Mkaif-Agb/Corona-Virus-Image-Detection/master/Healthy.png)
![Corona](https://raw.githubusercontent.com/Mkaif-Agb/Corona-Virus-Image-Detection/master/Corona.png)
