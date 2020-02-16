Implementation of SVM classifier from scratch (without packages usage)
The algorithm updating is based onstochastic gradient descent (SGD)

The code contain:
comparison of my function to thre built-in sklearn SVM function
 
![Q2_validate](https://user-images.githubusercontent.com/53649764/74610552-e7410480-50fc-11ea-881f-5352c5cdb640.png)


Classification of the Iris dataset with different 𝜆 hyperparameter (all with Linear Kernel)

The meaning of 𝜆 hyperparameter - In SVM loss function, the 𝜆 parameter is multiplied by the weights vector 
This influences the outcome in two ways:
1)	Weights – As we increase 𝜆, the model is more “severely punished” for increasing weight size
2)	Margins – As we increase 𝜆, the model will provide a separation plane with wider margins
Each problem and sample data set have different characteristics and therefore there is no strict rule to determine an optimal 𝜆 value – it has to be tuned

In our case, 5 different 𝜆 were tested. Examining test accuracy (test error) it seems that the two models perform best equally, when 𝜆 = 0 and 𝜆 = 0.05. for these 𝜆 values, the model gets a test error of 0.033. Of these two models the one that has 𝜆 = 0.05, has the wider margins (0.8), which indicates the minimal distance between the correctly classified train samples and the separating plane. Higher margin value (higher bias) is an indication that the model might generalize better to real world samples, therefore the best choice is the 𝜆 = 0.05 model. 

![Q2eErrors](https://user-images.githubusercontent.com/53649764/74610554-e8723180-50fc-11ea-93c0-73dcf9da3642.png)

![Q2eMergins](https://user-images.githubusercontent.com/53649764/74610555-e90ac800-50fc-11ea-8df1-69f3fe8bfa7f.png)
