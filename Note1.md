# Lecture 2

## Some challanges in CV
1. Viewpoint Variation
2. Illumination
3. Deformation
4. Occlusion
5. Background Clutter
6. Intraclass Variation(species)

> We use data-driven approach.
``` python

def train(images, labels):
  return model
  
def predict(model, test_images):
  return test_labels


```
## Nearest Neighbor Model

Drawback: O(1) -> fast in training,
          O(N) -> slow in predicting.
          
## K Means Model
Instead of copying label from nearest neighbor, take majority vote from k nearest neighbors. 
K hyperparameters, can't be learnt from data directly 

## Correct way to test hyperparameters
* Set three sets: training, validation, tests
* train the data on the training set
* validate it on the validation set
* then choose the one who performs best on the validation set
* test the model on test set and record the result

## cross validation
* Split data into n folds
* train on n - 1 fold and validate on 1 fold, and then pick another fold until all fold is choosed
* Not very useful in deep learning

## Drawbacks of K means:
* Slow
* Curse of dimensionality
* Need lots of data so that they can roughly take up the sample space

## Linear Classifier:
f(x, W) = Wx + b

# Lecture 3

## Loss function
A loss function tells how good our classifier is

<a href="https://www.codecogs.com/eqnedit.php?latex=L&space;=&space;\frac{1}{N}\sum(L_i(f(X_i,&space;W),&space;y_i)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L&space;=&space;\frac{1}{N}\sum(L_i(f(X_i,&space;W),&space;y_i)))" title="L = \frac{1}{N}\sum(L_i(f(X_i, W), y_i)))" /></a>

## SVM

### Max/Min possible loss function
Max: Positive Infinity
Min: 0

### If we initialize with all zero, what will be the loss at the very beginning?
Denote the number of classes as C, then the loss should be C - 1(useful when you debug your program).

### Code Example
<a href="https://www.codecogs.com/eqnedit.php?latex=L_{i}=\sum_{j\neq&space;y_{i}}max(0,&space;s_{j}&space;-&space;s_{y_{i}}&space;&plus;&space;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L_{i}=\sum_{j\neq&space;y_{i}}max(0,&space;s_{j}&space;-&space;s_{y_{i}}&space;&plus;&space;1)" title="L_{i}=\sum_{j\neq y_{i}}max(0, s_{j} - s_{y_{i}} + 1)" /></a>

Note: As I understand, the following function only calculates loss for one element, which means, only one column.

``` python 
def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  scores = W.dot(x)
  # compute the margins for all classes in one vector operation
  margins = np.maximum(0, scores - scores[y] + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i
  

```
The optimal W to make loss equals zero is not unique.


## Regularization
Some example: L1, L2, Elastic Net(L1 + L2), Dropout

## Softmax Classifier
scores = unnormalized log probabilities of the class
In summary,
<a href="https://www.codecogs.com/eqnedit.php?latex=L_{i}&space;=&space;-log(\frac{e^{s_{y_i}}}{\sum_{j}(e^{s_{j}})})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L_{i}&space;=&space;-log(\frac{e^{s_{y_i}}}{\sum_{j}(e^{s_{j}})})" title="L_{i} = -log(\frac{e^{s_{y_i}}}{\sum_{j}(e^{s_{j}})})" /></a>


## Optimization

### Gradient
Always use analytic gradient, and you can use numerical gradient when checking(gradient checking).

# Lecture 4

## Back Propogation
Why back propogation? More efficient than trivial derivative computing and forward-mode pass(Since forward mode computes multiple-to-one while back propogation computes one-to-multiple and that's what we need indeed).

In a very approximate sense, you can understand BP as reverse-mode chain rule. 
### An intuitive understanding of derivative
The derivative on each variable tells you the sensitivity of the whole expression on its value.

### Abstraction
When considering gate, if we know the expression of a derivative, we can compress lots of nodes into another functionalty node. 

### Some gates:
Add gate: Gradient distributor
Max gate: Gradient Router
Mul Gate: Gradient Switcher

# Lecture 5

## CNN History
Breakthrough: 2012, Alex Net

## Convolutoin and Pooling

### Difference between convolutional layers
> Preserve Spacial Structure
What do convolutioin means?
For example, for a 32 * 3 * 3 photo, we use a 5 * 5 * 3 filter, slide over the whole matrix, and do dot products.
Notes:
1. The depths are always the same(the third dimension)

### Stride
Size of the output layer: (N - F) / stride + 1
N: size of the image, F: size of the filter

In practice, we often zero pad the border with magnitude (F - 1) / 2.

### Pooling Layer
* Make the representations smaller and more managable
* Operates over each activation map independently
* Have nothing to do with the depth
In practice, we often use max pooling. 
