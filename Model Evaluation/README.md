**Models Evaluation With Cross Validation**

Running SVM models with different Kernels and 𝜆 to choose best model for the specific problem\
In this case classifing the MNIST data set
For each model we held cross validation to evaluate it preformance on the data,\
evaluation was based on the average train error and average test error

Test error gets the minimum value when using the rbf kernel model with 𝜆 hyperparameter equal to 0.01.\
We can see improvement of test error with the increase in model complexity until a point where the model is getting too complex and the outcome is overfitting:\
•	In the polynomial kernel models, we can see that the train error decreases with every increase of the polynomial degree which eventually leads to overfitting. We can see that with 10th degree the train error still improving while test error starts rising.\
•	A similar effect is happening in the rbf kernel models with more extreme manifestation. Train and test error improve with bigger values of gamma but for gamma bigger than 0.01 test error results increase dramatically with a distinct indication of overfitting.
	
We could predict that the best model would be the rbf kernel with 𝜆 value of 0.01 because validation error is expected to act the same way as the test error, especially when cross validation was preformed and therefore validation is less depended on the random split of the data.
The described above effect of decreasing error with rise of model complexity until the model reaches the complexity limit and start overfitting can be seen also in the validation results. Accordingly, we could guess that the model with lowest validation error and lowest train error (but not zero) will be perform best on the test data.

The runing of this script is ~1hr long,\
Dictionary of the result is inside the code and commented

![Q4_](https://user-images.githubusercontent.com/53649764/75095630-91a6a500-559f-11ea-92ec-8c56a5f616a6.png)
