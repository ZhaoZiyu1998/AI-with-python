import numpy as np

### simple neural network ###
input=[1.2,5.1,2.1]
weights=[3.2,2.8,9.4]
bias=3
output= input[0]*weights[0] + input[1]*weights[1] + input[2]*weights[2] + bias
print(output)
### 4 inputs and 3 neurons ###
input=[1.2,5.1,2.1,3.1]
weight1=[3.2,2.8,9.4,1.2]
weight2=[2.1,8.1,9.1,1.1]
weight3=[1.2,5.1,2.1,3.1]
bias1=3
bias2=2
bias3=7
output1= input[0]*weight1[0] + input[1]*weight1[1] + input[2]*weight1[2] + input[3]*weight1[3] + bias1
output2= input[0]*weight2[0] + input[1]*weight2[1] + input[2]*weight2[2] + input[3]*weight2[3] + bias2
output3= input[0]*weight3[0] + input[1]*weight3[1] + input[2]*weight3[2] + input[3]*weight3[3] + bias3
output=[output1,output2,output3]
#### do it with numpy ###
input=[1.2,5.1,2.1,3.1]
weights=[[3.2,2.8,9.4,1.2],[2.1,8.1,9.1,1.1],[1.2,5.1,2.1,3.1]]
biases=[3,2,7]
output=np.dot(weights,input) + biases
print(output)

### Part 4 ###
