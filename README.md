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

**Data drift vs Concept drift**

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

- soon  ...

**Xavier Initialization vs He Initialization**
 - soon  ...


**L1 vs L2 Regularization**

 - soon  ...

**Model Hyper-parameters Optimization**
 - Grid Search
 - Random Search
 - Bayesian Search

**Time Series Models**

 - ARIMA
 - [BiLSTM](https://www.baeldung.com/cs/bidirectional-vs-unidirectional-lstm)
 
    <img src="./img/lstm.png" width="650">
 
 - CNN-LSTM
 - Variational autoencoder
 - Temporal Fusion Transformers 
 - Fourier Transform
 
 
 **Dependent and Independent Variables**
 - The input variables are known as the independent variables, as each variable must describe a distinct aspect of the data which is not available in others.
 - The target variable is known as a dependent variable as the changes in input values will affect the target value.  
 
 
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
  
  
  
 
 
 




