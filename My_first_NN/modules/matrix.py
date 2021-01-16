import random
import numpy as np


class Matrix:
    def __init__(self,rows,cols):
        self.rows = rows
        self.cols = cols
        self.data = []
        self.randomize()
    
    def randomize(self):
        self.data = [[random.randrange(11) for i in range(self.cols)]for j in range(self.rows)] 
        
    
    def print(self):
        print(np.matrix(self.data))

    def scalar_product(self,n):
         for i in range(self.rows):
             for j in range(self.cols):
                 self.data[i][j] *= n
    
    def scalar_add(self,n):
        for i in range(self.rows):
             for j in range(self.cols):
                 self.data[i][j] += n

    def map(self,fun):
        for i in range(self.rows):
             for j in range(self.cols):
                 self.data[i][j]  = fun(self.data[i][j])
    @staticmethod
    def static_map(m,fun):
        for i in range(m.rows):
             for j in range(m.cols):
                 m.data[i][j]  = fun(m.data[i][j])
        return m
    
    @staticmethod
    def transpose(m1):
        m2 = Matrix(m1.cols,m1.rows)
        for i in range(m1.rows):
            for j in range(m1.cols):
                m2.data[j][i] = m1.data[i][j]
        return m2    

    @staticmethod
    def add(m1,m2):
        assert m1.rows==m2.rows and m1.cols==m2.cols,'Impossible do addition'
        m3 = Matrix(m1.rows,m2.cols)
        for i in range(m3.rows):
            for j in range(m3.cols):
                m3.data[i][j] = m1.data[i][j] + m2.data[i][j]
        return m3
    
    @staticmethod
    def multiply(m1,m2):
        assert m1.cols==m2.rows,'Impossible do multiplication'
        m3 = Matrix(m1.rows,m2.cols)
        for i in range(m1.rows):
            for j in range(m2.cols):
                s=0
                for k in range(m1.cols):
                    s += m1.data[i][k]*m2.data[k][j]
                m3.data[i][j] = s
        return m3
    
    @staticmethod
    def fromArray(arr):
        m1 = Matrix(len(arr),1)
        for i in range(m1.rows):
           for j in range(m1.cols):
               m1.data[i][j] = arr[i]               
        return m1

    @staticmethod
    def toArray(m1):
        arr = []
        for i in range(m1.rows):
            for j in range(m1.cols):
                arr.append(m1.data[i][j])
        return arr
    
    @staticmethod
    def difference(m1,m2):
        assert m1.rows==m2.rows and m1.cols==m2.cols,'Impossible do substraction'
        m3 = Matrix(m1.rows,m2.cols)
        for i in range(m1.rows):
            for j in range(m1.cols):
                m3.data[i][j] = m1.data[i][j] - m2.data[i][j]
        return m3
    
    @staticmethod
    def from_array_to_matrix(arr):
        rows = len(arr)
        cols = 1
        m = Matrix(rows,cols)
        for i in range(rows):
            for j in range(cols):
                m.data[i][j] = arr[i]
        return m
    
    @staticmethod
    def multiply_element(m1,m2):
        m3 = Matrix(m1.rows,m2.cols)
        for i in range(m3.rows):
            for j in range(m3.cols):
                m3.data[i][j] = m1.data[i][j]*m2.data[i][j]
        return m3
    
