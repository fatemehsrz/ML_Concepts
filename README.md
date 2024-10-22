# Machine Learning Essential Concepts

**Model Calibration**

- A calibrated model produces calibrated probabilities which means a prediction of a class with confidence p is correct 100*p percent of the time. Consider this example: if a model trained to classify images as either containing a cat or not, is presented with 10 pictures, and outputs the probability of there being a cat as 0.6 (or 60%) for every image, we expect 6 cat images to be present in the set. With calibrated probabilities, we might directly interpret the produced numbers as the confidence of the model. When encountering easy samples from the dataset, for which the model is rarely wrong, we expect the model to produce values close to 1. For harder samples, we expect the number to be lower, reflecting the proportion of misclassified examples in such sets. [More here](https://www.unofficialgoogledatascience.com/2021/04/why-model-calibration-matters-and-how.html).

<img src="./img/blog.png" width="450">

**Model Overfitting vs Underfitting**

 - soon ...

**Concept drift vs Model drift**

**Gradient Exploding vs Gradient Vanishing**

- soon  ...

**Momentum in Neural Networks**

**Xavier Initialization vs He Initialization**

**Model Hyper-parameters Optimization**
 - Grid Search
 - Random Search
 - Bayesian Search


**Data Interpolation**

- soon ...

**Data Augmentation**

- soon ....


**plobplot and lmplot**

**SMOTE Oversampling**

**Clustering**
  - Spectral clustering
  - Hierarchical clustering

**Likelihood vs probability**

**Prior Probability vs Posterior Probability vs Likelihood**

**SVD vs Eigen Decomposition**

**Loss Functions**

  - Hinge loss
  - KL loss
  - Cross entropy loss
  - Huber loss
 
 
 **Boosting vs Bagging**
 
 **XGBoost vs Catboost**
 
 
 **High Bias vs High Variance**
 
 **Convex vs non-Convex Optimization Problem**
 
 **Statistical Concepts**
 
 - Significant test
 - AB Testing
 - ANOVA
 - Information gain
 - Surprise value and Entropy value
 
 **Distributions**
 
  - Bernoulli Distribution
  - Binomial Distribution
  - Multinomial distribution
  - Poisson Distribution
  - Chi-squared Distribution
  - Gamma distribution
  
  
  
 
 
 




