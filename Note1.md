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
$f(x, W) = Wx + b$


