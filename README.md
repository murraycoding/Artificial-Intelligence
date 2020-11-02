# Machine Learning Fundamentials

## Artificial Intelligence vs Netual Networks vs Machine Learning

### What is artificial intelligence
The effort to automate intellectual tasks normally performed by humans. This can be just a simple set of instructions for a computer to follow. Technically, even something as simple as a computer player for tic-tac-toe. 

## What is machine learning?
Machine learning is a subset of artificial intelligence. In typical simple AI, the computer is given all of the rules of the game (if it is a game being played). In Machine learning, the goal is to give the program information and for the computer to figure out the best moves based on the desired outcome. In the case of the game we would need to tell the computer how to win the game. 

This also requires some training data so the computer knows what are examples of good moves and bad moves. In summary, machine learning finds the rules for us, rather than us giving the program the rules. 

## Neural Networks (Deep learning)
This is a further subset of machine learning which uses a layered representation of data. In neural networks, you can give the program as many input layers as you would like, and then it takes all of the input variables and computes a result. Similar to general machine learning. You need to give the NN training data along with a desired outcome. This data is used to train the comprehension layer which takes the input data and produces an outcome.

Multi-stage information extraction?? This is the name of the differeny computation layers.

## Machine LEarning Terminology

### Features / Labels
Features are the information given to the machine learning model. The label is the information we are trying to predict. When training the model, we also need to give the model the correct label compared to the predicted label. When using the completed model, you only have the features and you are trying to correctly predict the label.

## Types of Machine Learning

### Unsupervied Learning
This form of machine learning is the process of taking features only and creating similar groups of data which are similar. 

### Supervised Learning
Supervised learning is the process of training a model with a set of features and labels and using the model to predict future labels. Within this process, you need to test the model to see how accurate it is. Once the model is at a point to successfully predict labels given new features, you can stop training the model.Most machine learning algorithms are based on supervised learning.

### Reinforcement Learning
You do not start any data but you start with an agent, environment and a reward. This is most common when training a AI to play a video game. The agent in this case the player. The environment the playable area. The reward would be the computer rewarding the agent closer to the end of the game.

# Introduction to Tensor Flow

## Intro to TensorFlow

### What is TensorFlow
This is an open-source library for machine learning maintained by Google. You can do image classification, data clustering, regression, reienforment learning and natural language processing. Most of the complicated math is done by the library and not by the user. 

The library is powered by "graphs" and "sessions". Graphs are created by our code we write. For example, if we define both variable 1 and variable two and also define that we would like the program to find the sum of variable 1 variable 2, this is all part of the graph. In the graph, we just have the equation and not the answer to the equation. This is called a graph becuase several computation can be linked together. 

A "session" happens when the equations and variables in the graph start to be processed. The session will start at the end of the graph which is not dependent on other things (such as constant values and input from outside sources - not other equations). Then, the program will work through some of the more connected parts of the graph until the session produces a result. 

### Installing/Importing TensorFlow

## Tensors

### What are Tensors?
A vector generalize for higher dimensions. Each tensor represents a partially defined computation which will eventually produce a value. The session will execute different tensors as needed. Each tensor has a data type and a shape. The tensor "types" are defined by what type of data is sorted within them. The shape is the dimension of the vector.


### Creating Tensors
To create a tensor, you can use the following commands. These are all scalar vectors with different data types since they are just one number. The command to create a tensor is the code you see below with the values followed by the data type of the values.

  string = tf.Variable("this is a string", tf.string)
  number = tf.Variable(324, tf.int16)
  floating = tf.Variable(3.567, tf.float64)

### Rank/Degree of Tensors
The rank of the each tensor describes the dimensions of the vector. A scalar vector has rank 0 since it is not even part of a list. See below for each type of rank of each of the tensors. Another way to think of the rank is the amount of nested levels there are in the list (which defines the tensor). In 'rank2_tensor' there is a list within another list (2 levels) so it is considered rank 2.

  tensorA = tf.Variable("Brian", tf.string)
  tensorB = tf.Variable(["Brian"], tf.string)
  tensorC = tf.Variable([["Brian","Taylor"],["Murray","Murray"]], tf.string)
  
To determine the rank of the tensor, you can use the command tf.rank. The examples below print the rank of the above tensors.

  # print out the rank of each tensor
  print(f'tensorA: Rank {tf.rank(tensorA)}')
  print(f'tensorB: Rank {tf.rank(tensorB)}')
  print(f'tensorC: Rank {tf.rank(tensorC)}')
  
  # Output
  tensorA: Rank 0
  tensorB: Rank 1
  tensorC: Rank 2

### Shape of Tensors
The 'shape' of the tensor gives the size of the matrix it would represent. For example, a rank 2 vector with 2 values in each list is just a 2x2 matrix so the shape would be [2,2]. The examples below gives the shape of each tensor defined above.

  # finds the shape of each of the tensors
  print(f'tensorA: Shape {tf.shape(tensorA)}')
  print(f'tensorB: Shape {tf.shape(tensorB)}')
  print(f'tensorC: Shape {tf.shape(tensorC)}')
  
  # Output
  tensorA: Shape []
  tensorB: Shape [1]
  tensorC: Shape [2 2]
  
### Chaning Shape
Similar to NumPy, given an initial array of values, you can change the shape of the array given all of the elements fit into uniform rows and columns. For example, a tensor of rank 2 and shape [4 2] has a total of 8 elements so it would not fit into a tensor of shape [3 3] since the new tensor would need 9 elements.

### Slicing Tensors

### Types of Tensors

# Core Learning Algorithms

# Neural Networks with TensorFlow

# Deep Computer Vision

# Natural Language Processing

# Reinforcment Learning
