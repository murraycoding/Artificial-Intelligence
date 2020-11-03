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
Similar to NumPy, given an initial array of values, you can change the shape of the array given all of the elements fit into uniform rows and columns. For example, a tensor of rank 2 and shape [4 2] has a total of 8 elements so it would not fit into a tensor of shape [3 3] since the new tensor would need 9 elements. In the example below, we start with an array which has a total of 24 elements (shape: [4 2 3]) and then reshape the tensor in different ways. In the third example there is a -1 in place of where a typical number should be. The negative one tells the program to find the number that should fix there given one can exist. In the example we are asking for the program to find an array size 3 by something which will hold the same 24 elements. Since 3x8=24, then the program will create a tensor of shape [3 8].

    tensorD1 = tf.ones([4,2,3])  # 4x2x3 = 24
    tensorD2 = tf.reshape(tensorD1,[2,2,6])  # 2x2x6 = 24
    tensorD3 = tf.reshape(tensorD2, [3,-1]) # the -1 tells the program to just find the correct shape
    
    # prints the vectors
    print(tensorD1)
    print(tensorD2)
    print(tensorD3) 
    
    tf.Tensor(
    [[[1. 1. 1.]
      [1. 1. 1.]]

     [[1. 1. 1.]
      [1. 1. 1.]]

     [[1. 1. 1.]
      [1. 1. 1.]]

     [[1. 1. 1.]
      [1. 1. 1.]]], shape=(4, 2, 3), dtype=float32)
    tf.Tensor(
    [[[1. 1. 1. 1. 1. 1.]
      [1. 1. 1. 1. 1. 1.]]

     [[1. 1. 1. 1. 1. 1.]
      [1. 1. 1. 1. 1. 1.]]], shape=(2, 2, 6), dtype=float32)
    tf.Tensor(
    [[1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1.]], shape=(3, 8), dtype=float32)
     
### Slicing Tensors
No notes?

### Types of Tensors
With the exception of variable, all of the tensors below are immutable (the value cannot change).

#### Variable


#### Constant


#### Placeholder


#### SparseTensor




### Evaluating Tensors
To evaluate a tensor, you need to create a session. The default code for this is very similar to the code for opening a file and reading information form it.

    with tf.Session() as sess: # creates a session using the default graph
        my_tensor.eval()       # my_tensor will be the name of the tensor



# Core Learning Algorithms

## Linear Regression

### Preparing the data
Before running any sort of machine learning code, you need to tell the program about the type of data it will be seeing. In the example in the video (the same as the example from the tensorflow website), we are working with data about the passengers aboard the Titanic. The code below loads the dataset using pandas and then removes the 'survived' column since this is what we are trying to predict.

    # Load dataset.
    dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
    dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')

In the example, there are two types of data. numerical data and categorical data. In the next lines of code, the names of the categorical data values are given and all possible values of each feature are put into a list. By giving the computer a list of all of the possible values, the library will automatically convert all of the categorical data into numberic data for the program to calculate predictions with. 

    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                           'embark_town', 'alone']
    NUMERIC_COLUMNS = ['age', 'fare']

    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
      vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
      feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
      feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    print(feature_columns)
    
### Feeding the model
The next step in the process is to feed the training data to the model. Often times in machine learning there will be potentially millions of training data points to feed a model with. Due to this large amount of data, the model needs to be given information in batches. The best case scenario for loading data is to load about 32 data points at a time. If you were to load 1 piece of data at a time, it would take too long and be too inefficient. If you load too much, then the memory of the computer would become too full and overload. 

### Epochs
How many times the model will see the same model. The first time the model sees the data. It will produce a result which will be relatively inaccurately. The next time the data is given to the model, the data will be given in a different order. Seeing the same data in different ways (or orders) make it easier for the model to pick up on patterns. An Epoch is just one stream of the entire dataset. While it is good for the model to see the data multiple times, there is such a thing as overfitting. Overfitting is where the model becomes too relient on the dataset and essentially memorizes the dataset. The goal of all machine learning is to train the computer to take in new data values and do something with that. Having a model which is too relient on specific data can lead to innacuurate predictions on any future data.

### Input function
An input function defines how the data will be broken into batches (pieces of data to feed to the function) and epochs (the number of times the model will see the data). In Tensorflow, the model needs to be given the data in a very specific type of object built into TensorFlow, the 'tf.data.DataSet' object. The input function will take the data from a pandas dataframe and convert it into the DataSet object in order to give it to the model.

The code for the input function is rather complicated but it really comes down to a function creating the input functions. This is coded this way to allow for a function to produce a separate function for the training data and another function for the evaluation data.

    (1)  def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    (2)      def input_function():  # inner function, this will be returned
    (3)          ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    (4)          if shuffle:
    (5)              ds = ds.shuffle(1000)  # randomize order of data
    (6)          ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    (7)          return ds  # return a batch of the dataset
    (8)      return input_function  # return a function object for use

    (9)  train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
    (10) eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

Here is a line by line breakdown of the code below:
Line (1) is the outer function which will return the different input functions. There are two required parameters the data (features) and the label data (the correct results of what the model is trying to predict. The reason only the data is required and not the other values is because by default, you would want to train the data with around 10 epochs (you can always play with this number to tweak the model later) and you should always shuffle the data before starting the next epoch. 
Line (2) is the definition of the input function, this function has no parameters since all of the needed parameters are defined in the outer function definition. 
Line (3) This is the tensorflow method to take the data in the form of pandas dataframes and convert it to a Dataset object needed for the model. The dict() method is to cover the multi-column data to a dictionary.
Line (4) Only executes the code if shuffle is set to True (which is set by default)
Line (5) redefines the data set as a shuffled version of the dataset.
Line (6) The dataset is split into "batches" and repeats for the number of "epochs"
Line (7) The return statement for the innner function to return the dataset.
Line (8) The return statement for the outer function which will return the input function with all of the correct settings
Line (9) Defines the training input function as a generic input function with the training data and the predictions
Line (10) Defines the evaluation input function with the evaluation data and specifically sets the number of epochs to 1 and the shuffle to false since this the evaluation data and not the training data.


## Classification

## Clustering

## Hidden Markov Models

# Neural Networks with TensorFlow

# Deep Computer Vision

# Natural Language Processing

# Reinforcment Learning
