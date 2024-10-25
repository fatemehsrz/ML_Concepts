# Machine Learning Essential Concepts

**Model Calibration**

- A calibrated model produces calibrated probabilities which means a prediction of a class with confidence p is correct 100*p percent of the time.
 It is important to know which percentage of samples that model predicted to be positive (with p probability) are actually positive in the dataset. With a well calibrated model, we expect that p of the samples that got a predicted probability of around p to be positive. Model calibration measures how well the model prediction is aligned with true distribution of the data. [More here ...](https://wttech.blog/blog/2021/a-guide-to-model-calibration/).


<img src="./img/blog.png" width="430">

**Model Overfitting vs Underfitting**

 - Overfitting occurs when a model is too complex and fits the training data too closely. This leads to poor generalization.
Underfitting occurs when a machine learning model is too simple leading to low model accuracy and poor performance. [More here ...](https://databasecamp.de/en/ml/underfitting-en)

 - How to resolve overfitting:
   - Adding more data samples
   - Adding noise to the data
   - Data augmentation
   - Feature selection
   - Cross Validation
   - Model regularization (loss, or adding dropouts)
   
   
 - How to resolve underfitting:
   - Add more features (e.g. feature engineering)
   - Increase model complexity
   - Data denoising
   - Running more training epochs 
 
 
 
 <img src="./img/fit.png" width="680">

**Data Drift vs Concept Drift**

- Data drift is the changes over time in the statistical properties of the input data. It occurs in production  as the new incoming data deviates from the original data the model was trained on or earlier production data. This shift in input data distribution can lead to a decline in the model's performance and must be detected. Concept drift relates to changes occurring over time in the relationships between input and target variables. It can cause a decay in the model quality and lead to a poor estimation on target values. [More here ...](https://www.evidentlyai.com/ml-in-production/data-drift) 

   
   <img src="./img/drft.png" width="600">

**Gradient Exploding vs Gradient Vanishing**

 - Gradient vanishing occurs when model's weights become extremely small (close to zero) as they are backpropagated through the layers of neural networks. It means the model's weights are not updated effectively and cannot learn the complex pattern in the data. How to resolve gradient vanishing
 
   - Apply different activation functions (e.g. tanh, Sigmoid, ReLU, Maxout,  ELU, SoftPlus, softsign, seLU)
   - Try different weight initialization (Xavier or He initializer)
   - Replace different optimizer (Adam, SGD, RMSprop, AdamW, Adadelta)
   - Tune learning rate from `1-e5` to `0.1` 
 
 
 - Gradient exploding  occurs when the loss value grows exponentially during training. It cause the large update to the weights and weights become NaN or infinity. How to resolve gradient exploding
 
   - Use batch normalization
   - Decrease number of layers in the model
   - Try different weight initialization
   - Apply gradient clipping which restricts weights in a certain range 
   - [More here ...](https://aiml.com/what-do-you-mean-by-vanishing-and-exploding-gradient-problem-and-how-are-they-typically-addressed/)
 
  <img src="./img/gr2.png" width="700">

**Momentum in Neural Networks**

- During training, gradient descent does not exactly provide the direction in which the loss function is headed i.e. the derivative of the loss function. Therefore, the loss value might not always be headed in the optimal direction and can easily gets stuck in a local minima. To avoid this situation, we use a momentum term in the objective function, which is a value between `0` and `1` that increases the size of the steps taken towards the minimum by trying to jump from a local minima. If the momentum term is large then the learning rate should be kept smaller. A large value of momentum also means that the convergence will happen fast. But if both the momentum and learning rate are kept at large values, then you might skip the minimum with a huge step. A small value of momentum cannot reliably avoid local minima, and can also slow down the training of the system. Momentum also helps in smoothing out the variations, if the gradient keeps changing direction. A right value of momentum can be either learned by hit and trial or through cross-validation. [More here](https://medium.com/analytics-vidhya/momentum-a-simple-yet-efficient-optimizing-technique-ef76834e4423)

<img src="./img/grd1.jpg" width="650">

**Xavier Initialization vs He Initialization**

Xavier/Glorot initialization is designed to address the problem of vanishing or exploding gradients during training. It sets the initial weights in a way that helps stabilize the training process. With the random initialization we do not know anything about the data, so we are not sure how to assign the weights that would work in that particular case. One good way is to assign the weights from a Gaussian distribution. Obviously this distribution would have zero mean and some finite variance. Let’s consider a linear neuron:

$y = w1x1 + w2x2 + ... + wNxN + b$

With each passing layer, we want the variance to remain the same. This helps us keep the signal from exploding to a high value or vanishing to zero. In other words, we need to initialize the weights in such a way that the variance remains the same for x and y. This initialization process is known as Xavier initialization. 


Indeed, we want the variance to remain the same as we pass through each layer. Let’s go ahead and compute the variance of y:

$var(y) = var(w1x1 + w2x2 + ... + wNxN + b)$

Let’s compute the variance of the terms inside the parentheses on the right hand side of the above equation. If you consider a general term, we have:

$var(wixi) = E(xi)2var(wi) + E(wi)2var(xi) + var(wi)var(xi)$

Here, E() stands for expectation of a given variable, which basically represents the mean value. We have assumed that the inputs and weights are coming from a Gaussian distribution of zero mean. Hence the “E()” term vanishes and we get:

$var(wixi) = var(wi)var(xi)$

Note that ‘b’ is a constant and has zero variance, so it will vanish. Let’s substitute in the original equation:

$var(y) = var(w1)var(x1) + ... + var(wN)var(xN)$

Since they are all identically distributed, we can write:

$var(y) = N * var(wi) * var(xi)$

So if we want the variance of y to be the same as x, then the term `N * var(wi)` should be equal to `1`. Hence:

$N * var(wi) = 1$
$var(wi) = 1/N$

Now we arrived at the Xavier initialization formula. We need to pick the weights from a Gaussian distribution with zero mean and a variance of `1/N`, where `N` specifies the number of input neurons. So the formula becomes:

$var(wi) = 1/Navg $
where $Navg = (Nin + Nout)/2$

The reason they do this is to preserve the backpropagated signal as well. However, it is more computationally complex to implement. Hence we only take the number of input neurons during practical implementation.


**L1 vs L2 Regularization**

 - soon  ...

**Model Hyper-parameters Optimization**
 - Grid Search
 - Random Search
 - Bayesian Search
 
  <img src="./img/grid.jpg" width="650">

**Time Series Models**

 - ARIMA
 - [BiLSTM](https://www.baeldung.com/cs/bidirectional-vs-unidirectional-lstm)
 
    <img src="./img/lstm.png" width="650">
 
 - CNN-LSTM
 - Variational autoencoder
 - Temporal Fusion Transformers 
 - Fourier Transform
 
 
 **Dependent vs Independent Variables**
 - The input variables are known as the independent variables, as each variable must describe a distinct aspect of the data which is not available in others.
 - The target variable is known as a dependent variable as the changes in input values will affect the target value.  
 
     <img src="./img/cor2.png" width="650">
 
 
**Data Interpolation**

- soon ...

**Data Augmentation**

- soon ....

**SMOTE Oversampling**


**plobplot and lmplot**

- soon  ...

**Clustering**
  - Spectral clustering
  - Hierarchical clustering

**Likelihood vs probability**

- soon  ...

**Prior Probability vs Posterior Probability vs Likelihood**

- soon  ...

**SVD vs Eigen Decomposition**

- soon  ...

**Loss Functions**

  - Hinge loss
  - KL loss
  - Cross entropy loss
  - Huber loss
 
 
 **Boosting vs Bagging**
 
 - soon  ...
 
 **XGBoost vs Catboost**
 - soon  ...
 
 
 **High Bias vs High Variance**
 - soon  ...
 
 **Convex vs non-Convex Optimization Problem**
 - soon  ...
 
 **Statistical Concepts**
 
 - Significance test
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
  
  
  
 
 
 




