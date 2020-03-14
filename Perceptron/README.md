**Implementation of Perceptron binary classifier**\

The function gets:\
x - data matrix (can be in any dimension)\
y - data labels (0 or 1)

attached is an example on sklearn make_blobs dataset\
notice that data is generated  randomly so the result might be a little different if you will run the code\
I used random seed = 42\

As expected from Perceptron, the separator line is not the optimal line (maximal margins between the data groups)\
rather it is the first found by the algorithm that creates a full separation between the two groups,\
i.e. all the labels from the first group are located on one side of the line\
and all the labels from the second group are located on the second side of the separate line.\

![הורדה](https://user-images.githubusercontent.com/53649764/76687583-e9c84880-662d-11ea-9ecd-e4c76ff91706.png)
 
