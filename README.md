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

**Concept drift vs Model drift**

- Data drift is the changes over time in the statistical properties of the input data. It occurs in production  as the new incoming data deviates from the original data the model was trained on or earlier production data. This shift in input data distribution can lead to a decline in the model's performance and must be detected. Concept drift relates to changes occurring over time in the relationships between input and target variables. It can cause a decay in the model quality and lead to a poor estimation on target values. [More here ...](https://www.evidentlyai.com/ml-in-production/data-drift) 

   
   <img src="./img/drft.png" width="600">

**Gradient Exploding vs Gradient Vanishing**

- soon  ...

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
  
  
  
 
 
 




