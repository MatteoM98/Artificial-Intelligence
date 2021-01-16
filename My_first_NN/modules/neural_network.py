from .matrix import Matrix
from math import e
import random

def sigmoid(x):
    return 1 / (1+e**-x)

def dsigmoid(y):
    #return sigmoid(x) * (1-sigmoid(x)), but the function is yet applicated in the Matrix
    return y*(1-y)




class NeuralNetwork():
    def __init__(self,input_nodes,hidden_nodes,output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        #weight matrice nHidden x nInput
        self.weight_matrix_ih = Matrix(hidden_nodes,input_nodes)
        NeuralNetwork.initialize_weight_matrix(self.weight_matrix_ih)
        #weight matrice nOutput x nHidden
        self.weight_matrix_ho = Matrix(output_nodes,hidden_nodes)
        NeuralNetwork.initialize_weight_matrix(self.weight_matrix_ho)
        #bias matrices
        self.bias_matrix_h = Matrix(hidden_nodes,1)
        NeuralNetwork.initialize_weight_matrix(self.bias_matrix_h)
        self.bias_matrix_o = Matrix(output_nodes,1)
        NeuralNetwork.initialize_weight_matrix(self.bias_matrix_o)
        #initial learning rate
        self.learning_rate = 0.1
        self.hidden = None

    @staticmethod   #m is a Matrix object
    def initialize_weight_matrix(m):
        for i in range(m.rows):
            for j in range(m.cols):
                m.data[i][j] = random.uniform(-1,1)
        

    def print_weights_matrices(self):
        print('Matrix Weights Input-Hidden: ')
        self.weight_matrix_ih.print()
        print('\nMatrix Weights Hidden-Output: ')
        self.weight_matrix_ho.print()
    
    def print_bias_matrices(self):
        print('Matrix Bias Hidden: ')
        self.bias_matrix_h.print()
        print('\nMatrix Bias Output: ')
        self.bias_matrix_o.print()
    
    def feedforward(self,inputArr):
        #generating Hidden's outputs
        mi = Matrix.fromArray(inputArr)
        h1 = Matrix.multiply(self.weight_matrix_ih,mi)  #input x weights
        hidden = Matrix.add(h1,self.bias_matrix_h)      #bias + (inputs x weights)
        #activation function
        hidden.map(sigmoid)
        self.hidden = hidden

        #generating Outputs' outputs
        o1 = Matrix.multiply(self.weight_matrix_ho,hidden)  #hidden's output x weights
        output = Matrix.add(o1,self.bias_matrix_o)          #bias + (hidden's output x weights)
        #activation function
        output.map(sigmoid)

        return output
    
    def train(self,inputArr,targets):
        inputs = Matrix.from_array_to_matrix(inputArr)
        guess = self.feedforward(inputArr)
        
        #calcolate the Error: Targets - Guess
        targets_matrix = Matrix.from_array_to_matrix(targets)
        errors = Matrix.difference(targets_matrix,guess)

        #gradient
        m1 = Matrix.static_map(guess,dsigmoid)
        gradients = Matrix.multiply(m1,Matrix.transpose(errors))
        gradients.scalar_product(self.learning_rate)


        #hidden deltas 
        hidden_T = Matrix.transpose(self.hidden)
        weight_matrix_ho_deltas = Matrix.multiply(gradients,hidden_T)
        self.weight_matrix_ho = Matrix.add(self.weight_matrix_ho,weight_matrix_ho_deltas)

        #bias
        self.bias_matrix_o = Matrix.add(self.bias_matrix_o,gradients)

        #transpose the weight matrix ho, and calculate hidden layer error
        weight_matrix_ho_T = Matrix.transpose(self.weight_matrix_ho)
        hidden_errors = Matrix.multiply(weight_matrix_ho_T,errors)

        #calculate hidden gradient
        m3 = Matrix.static_map(self.hidden,dsigmoid)
        hidden_gradient = Matrix.multiply_element(m3,hidden_errors)
        hidden_gradient.scalar_product(self.learning_rate)

        #input deltas
        inputs_T = Matrix.transpose(inputs)
        weight_matrix_ih_deltas = Matrix.multiply(hidden_gradient,inputs_T)
        self.weight_matrix_ih = Matrix.add(self.weight_matrix_ih,weight_matrix_ih_deltas)

        #bias 
        self.bias_matrix_h = Matrix.add(self.bias_matrix_h,hidden_gradient)






    

        

