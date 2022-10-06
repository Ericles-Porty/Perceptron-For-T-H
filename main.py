import numpy as np

#T = [[1,1,1],[0,1,0],[0,1,0]]
#H = [[1,0,1],[1,1,1],[1,0,1]]

SIZE = 3 # Size of the dataset
N = SIZE * SIZE #Number of attributes

bias = 1 # Bias
treshold = 0  # Limiar
learning_factor = 0.4  # Learning Factor

# Input
T = np.asarray([[1, 1, 1], [0, 1, 0], [0, 1, 0]])
H = np.asarray([[1, 0, 1], [1, 1, 1], [1, 0, 1]])

# Output
target = [1, -1]

instances = []
weights = []

weights.append(bias)

# Input T in instances
instances.append([i for element in T for i in element])
instances[0].insert(0, bias)

# Input H in instances
instances.append([i for element in H for i in element])
instances[1].insert(0, bias)

def main():
    start_weights()

    train()
    
    # Test T
    test([1, 1, 1, 0, 1, 0, 0, 1, 0])
    # Test H
    test([1, 0, 1, 1, 1, 1, 1, 0, 1])
    # Test ??
    test([1, 1, 0, 1, 0, 1, 1, 1, 1])


# Initialize the random weights
def start_weights():
    for _ in range(N):
        weights.append(np.random.uniform(-1.0, 1.0))


# Activation Function
def signal(number):
    if number >= treshold:
        return 1
    else:
        return -1


# Balance the weights
def balance_weights(dataset_index, outputed_signal):
    weights[0] = weights[0] + learning_factor * bias * (target[dataset_index] - outputed_signal)
    for i in range(1, N+1):
        weights[i] = weights[i] + learning_factor * instances[dataset_index][i] * (target[dataset_index] - outputed_signal)


# y saves the signal of the sum of each Xi*Wi
def calculate_y(x):
    v = bias * weights[0]
    for index in range(1, N+1):
        v += x[index] * weights[index]
    y = signal(v)
    return y


# Train the neuron
def train():
    epoch = len(instances)
    while epoch > 0:
        for dataset_index in range(len(instances)):
            y = calculate_y(instances[dataset_index])
            if y == target[dataset_index]:
                epoch -= 1
            else:
                print("Balancing the weights!")
                balance_weights(dataset_index, y)
                epoch = len(instances)
    print("Converged\n")


# Test the weights
def test(dataset_test):
    dataset_test.insert(0, bias)
    y = calculate_y(dataset_test)
    print(dataset_test)
    if y == 1:
        print("Test result: T")
    else:
        print("Test result: H")
    print()

if __name__ == "__main__":
    main()

