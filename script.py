from modules.matrix import Matrix
from modules.neural_network import NeuralNetwork
import modules.neural_network
import random

training_data = [
    {
        'inp': [0,0],
        'out': [1]
    },
    {
        'inp': [0,1],
        'out': [0]
    }, 
    {
        'inp': [0,1],
        'out': [0]
    }, 
    {
        'inp': [1,0],
        'out': [0]
    }, 
    {
        'inp': [1,1],
        'out': [1]
    },
    {
        'inp': [1,1],
        'out': [1]
    } 
]

nn = NeuralNetwork(2,2,1)


for i in range(500_000):
    el = random.choice(training_data)
    inp = el['inp']
    out = el['out']
    nn.train(inp,out)

print('O xor 0 (1): ')
nn.feedforward([0,0]).print()
print('\n')

print('O xor 1 (0): ')
nn.feedforward([0,1]).print()
print('\n')

print('1 xor 0 (0): ')
nn.feedforward([1,0]).print()
print('\n')

print('1 xor 1 (1): ')
nn.feedforward([1,1]).print()
print('\n')


    