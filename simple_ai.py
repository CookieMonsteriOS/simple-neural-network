import random
import math

class SimpleNeuralNetwork:
    def __init__(self, input_size=2, hidden_size=3, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(hidden_size)] 
        self.b1 = [0] * hidden_size  
        self.W2 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(output_size)]  
        self.b2 = [0] * output_size


    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # Derivative of the sigmoid function - squash values between 0 -1
    def sigmoid_derivative(self, x):
        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)


    def dot_product(self, matrix, vector):
        # Ensure vector is 1D
        if isinstance(vector[0], list):
            vector = [item[0] for item in vector]  # Flatten 2D vector to 1D
        
        result = []
    
        for row in matrix:
            # Check if the dimensions match
            if len(row) != len(vector):
                raise ValueError(f"Matrix row length {len(row)} doesn't match vector length {len(vector)}")

            sum_product = sum(row[i] * vector[i] for i in range(len(vector)))
            result.append([sum_product])

        return result

    def forward_pass(self, x):
        # Input to hidden layer (Z1 = W1 * x + b1)
        Z1 = self.dot_product(self.W1, x)  # Multiply W1 by x
        A1 = [[self.sigmoid(Z1[i][0])] for i in range(len(Z1))]  # Apply sigmoid activation to Z1

        # Hidden to output layer (Z2 = W2 * A1 + b2)
        Z2 = self.dot_product(self.W2, A1)  # Multiply W2 by A1
        A2 = [[self.sigmoid(Z2[i][0])] for i in range(len(Z2))]  # Apply sigmoid activation to Z2

        return A1, A2


    def calculate_loss(self,Y,A2):
        return (Y[0] - A2[0][0]) ** 2          

    # Compute loss
    def compute_loss(self, A2, Y):
        A2_flat = [a[0] for a in A2]

        if len(A2_flat) != len(Y):
            raise ValueError(f"Output size {len(A2_flat)} doesn't match target size {len(Y)}")
        return sum((a - y) ** 2 for a, y in zip(A2_flat, Y)) / len(A2_flat)    

    # Backward pass
    def backward_pass(self, x, A1, A2, y, learning_rate):
        if len(A2) != len(y):
            raise ValueError(f"Length of A2 ({len(A2)}) does not match length of y ({len(y)})")

        # Calculate dA2 (error for output layer) / Loss how badly did we do?
        dA2 = [A2[i][0] - y[i] for i in range(len(A2))] 

        # Calculate dZ2 (error for output layer)
        dZ2 = dA2 

        # Gradients for W2 and b2
        dW2 = [[A1[i][0] * dZ2[j] for i in range(len(A1))] for j in range(len(dZ2))]
        db2 = dZ2  # Gradient for b2 is just the error dZ2

        # Backpropagate to the hidden layer
        dA1 = [0] * len(A1)  # Initialize dA1 to match number of hidden neurons
        for i in range(len(A1)):
            dA1[i] = sum([dZ2[j] * self.W2[j][i] for j in range(len(dZ2))])  # Backpropagate the error to hidden layer

        # Compute the error dZ1 for hidden layer
        dZ1 = [dA1[i] * A1[i][0] * (1 - A1[i][0]) for i in range(len(A1))]  # Sigmoid derivative for hidden layer
        dW1 = [[x[j] * dZ1[i] for j in range(len(x))] for i in range(len(dZ1))]  # Gradient for W1
        db1 = dZ1  # Gradient for b1

        # Update weights and biases using the gradients
        self.b2 = [self.b2[i] - learning_rate * db2[i] for i in range(len(self.b2))]
        self.b1 = [self.b1[i] - learning_rate * db1[i] for i in range(len(self.b1))]

        # Update weights using gradients
        self.W2 = [[self.W2[i][j] - learning_rate * dW2[i][j] for j in range(len(self.W2[0]))] for i in range(len(self.W2))]
        self.W1 = [[self.W1[i][j] - learning_rate * dW1[i][j] for j in range(len(self.W1[0]))] for i in range(len(self.W1))]


    # train
    def train(self, X_train, Y_train, epochs=5000, learning_rate=0.1):
        # Ensure that input vectors have the correct length (matching the number of columns in W1)
        for x in X_train:
            if len(x) != len(self.W1[0]):
                raise ValueError(f"Each input vector x should have length {len(self.W1[0])}, but got {len(x)}")

        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X_train, Y_train):
                A1, A2 = self.forward_pass(x)
                
                loss = self.compute_loss(A2, y)
                total_loss += loss

                self.backward_pass(x, A1, A2, y, learning_rate)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X_train)}")

        print(f"Training complete. Final loss: {total_loss / len(X_train)}")


    # Predict 
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            A1,A2 = self.forward_pass(x)
            predictions.append(A2)
        return predictions

    # evaluate    
    def evaluate(self, X_test, Y_test):
        total_loss = 0
        for x, y in zip(X_test,Y_test):
            A1, A2 = self.forward_pass(x)
            loss = self.calculate_loss(y,A2)
            total_loss += loss      
        print(f'Loss is: {total_loss / len(X_test)}')
        return total_loss / len(X_test)

if __name__ == "__main__":

    X_train =  [[0.5, 0.5],  
                [0.3, 0.7], 
                [0.9, 0.1]]

    Y_train = [[0], [1], [1], [0]]  

    model = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1)

    model.train(X_train, Y_train, epochs=5000, learning_rate=0.1)

    model.evaluate(X_train, Y_train)

    X_test = [[0,0], [0,1], [1,0], [1,1]]
    predictions = model.predict(X_test)

    # Print predictions
    print("Predictions:", predictions)            
        






