from ANN_Project_Assets import Loading_Datasets,Test_Model
import numpy as np
import copy
import matplotlib.pyplot as plt

'''
# Author : Amir Sabbagh Ziarani
# Version: 2.0  (without batching)
# Goal   : Training network by the use of Stochastic Gradient Descent & Backpropagation technics ( Loop )
# Epochs : 5
'''

def sigmoid(input):
    return 1.0 / (1.0 + np.exp(-1.0 * input))


def label_to_vector(label, output_neurons):
    vector = np.zeros(output_neurons)
    vector[int(label)] = 1
    return vector


def cost_MSE(predictedOutput, labels):
    tmp = np.subtract(predictedOutput, np.array([label_to_vector(labels, len(predictedOutput[0]))]))
    return np.sum(np.matmul(tmp, tmp.T))


def dev_cost_MSE(predicted, labels):
    return 2 * np.subtract(predicted, labels)


def dev_sigmoid(input):
    return np.multiply(sigmoid(input), 1 - sigmoid(input))


def multiply_multiple(input):
    res = input[0]
    for i in range(1, len(input)):
        res = np.multiply(res, input[i])
    return res


def feed_forward(input, input_weights, HL1_weights, HL2_weights):
    # add bias
    bias = np.array([np.ones(1)])

    # adding bias to first layer
    input_plus_bias = np.concatenate((bias.T, np.array([input])), axis=1)

    z0 = np.matmul(input_plus_bias, input_weights)
    a0 = sigmoid(z0)
    HL1_neurons_plus_bias = np.concatenate((bias.T, a0), axis=1)

    z1 = np.matmul(HL1_neurons_plus_bias, HL1_weights)
    a1 = sigmoid(z1)
    HL2_neurons_plus_bias = np.concatenate((bias.T, a1), axis=1)

    z2 = np.matmul(HL2_neurons_plus_bias, HL2_weights)
    a2 = sigmoid(z2)
    output = a2

    predicted = a2.argmax(axis=1)
    return output, predicted, z0, a0, z1, a1, z2, a2


def update_weights(input_weights, HL1_weights, HL2_weights,
                   learning_rate, batch_size,
                   grad_input_HL1_weights, grad_HL1_HL2_weights, grad_HL2_output_weights):
    new_input_weights = np.subtract(input_weights, grad_input_HL1_weights * (learning_rate / batch_size))
    new_HL1_weights = np.subtract(HL1_weights, grad_HL1_HL2_weights * (learning_rate / batch_size))
    new_HL2_weights = np.subtract(HL2_weights, grad_HL2_output_weights * (learning_rate / batch_size))

    return new_input_weights, new_HL1_weights, new_HL2_weights


def dCost_dAl_1(dAl, al_1, zl, w):
    dAl_1 = np.zeros(len(al_1[0]))
    tmpDer = 0
    for k in range(len(dAl_1)):
        for j in range(len(dAl)):
            tmpDer += dAl[j] * dev_sigmoid(zl[0][j]) * w[k][j]
        dAl_1[k] = tmpDer
    return dAl_1


def dCost_dWl_1(dZl_1, al_1, weight):
    al_1_b = np.concatenate((np.array([np.ones(1)]).T, al_1), axis=1)
    W_l = copy.deepcopy(weight)
    for k in range(len(weight)):
        for j in range(len(weight[0])):
            W_l = dZl_1[0][j] * al_1_b[0][k]
    return W_l


def update_gradient_Backpropagation(input, input_weights, HL1_weights, HL2_weights,
                                    learning_rate, batch_size,
                                    grad_input_HL1_weights, grad_HL1_HL2_weights, grad_HL2_output_weights,
                                    label, predicted, z0, a0, z1, a1, z2, a2):
    predicted = predicted[0]
    #  102input----W0------>(Z0)>150>(A0)----W1------>(Z1)>60>(A1)----W2------>(Z2)>4>(A2)-----Cost

    # print(np.shape(predicted), np.shape(z0), np.shape(a0), np.shape(z1), np.shape(a1), np.shape(z2), np.shape(a2))

    dA2 = dev_cost_MSE(predicted, label)
    dZ2 = np.multiply(dA2, dev_sigmoid(z2))
    dW2 = dCost_dWl_1(dZ2, a1, HL2_weights)

    dA1 = dCost_dAl_1(dAl=dA2, al_1=a1, zl=z2, w=HL2_weights)
    dZ1 = np.multiply(dA1, dev_sigmoid(z1))
    dW1 = dCost_dWl_1(dZ1, a0, HL1_weights)

    dA0 = dCost_dAl_1(dAl=dA1, al_1=a0, zl=z1, w=HL1_weights)
    dZ0 = np.multiply(dA0, dev_sigmoid(z0))
    dW0 = dCost_dWl_1(dZ0, np.array([input]), input_weights)

    return dW0, dW1, dW2


def train_backpropagation200():
    # load data
    train_set_features, train_set_labels, test_set_features, test_set_labels = Loading_Datasets.loadData()

    # shuffle data
    shuffler = np.random.permutation(len(train_set_features))
    train_set_features = train_set_features[shuffler]
    train_set_labels = train_set_labels[shuffler]

    # partition data
    limit = 200
    train_set_features = train_set_features[:limit]
    train_set_labels = train_set_labels[:limit]

    # allocate and initialize Weights
    HL1_neurons = 150
    input_HL1_weights = np.random.normal(loc=0, scale=1, size=(train_set_features.shape[1] + 1, HL1_neurons))
    HL2_neurons = 60
    HL1_HL2_weights = np.random.normal(loc=0, scale=1, size=(HL1_neurons + 1, HL2_neurons))
    output_neurons = 4
    HL2_output_weights = np.random.normal(loc=0, scale=1, size=(HL2_neurons + 1, output_neurons))

    # setting batch , epoch and learning_rate
    batch_size = 10
    epochs = 5
    learning_rate = 1

    # costList for plotting
    costList = []

    # itareate epochs
    for i in range(0, epochs):
        cost = 0
        # shuffling train set
        shuffler = np.random.permutation(len(train_set_features))
        train_set_features = train_set_features[shuffler]
        train_set_labels = train_set_labels[shuffler]

        # batching train set
        batched_train_set_features = train_set_features.reshape(int(len(train_set_features) / batch_size), batch_size,
                                                                len(train_set_features[0]))
        batched_train_set_labels = train_set_labels.reshape(int(len(train_set_labels) / batch_size), batch_size)

        # iterate on batches
        for batch in range(len(batched_train_set_features)):
            # allocate weights grads
            grad_input_HL1_weights = np.zeros(shape=(train_set_features.shape[1] + 1, HL1_neurons))
            grad_HL1_HL2_weights = np.zeros(shape=(HL1_neurons + 1, HL2_neurons))
            grad_HL2_output_weights = np.zeros(shape=(HL2_neurons + 1, output_neurons))

            # iterate on each batch images
            for image in range(len(batched_train_set_features[batch])):
                print("-v2------Epoch---- :", i, "------Image--- :", (batch * batch_size) + image)
                # computing output for this(current) image
                output, predicted, z0, a0, z1, a1, z2, a2 = feed_forward(batched_train_set_features[batch][image],
                                                                         input_HL1_weights, HL1_HL2_weights,
                                                                         HL2_output_weights)
                # update grads
                tmpG1, tmpG2, tmpG3 = update_gradient_Backpropagation(batched_train_set_features[batch][image],
                                                                      input_HL1_weights, HL1_HL2_weights,
                                                                      HL2_output_weights, learning_rate, batch_size,
                                                                      grad_input_HL1_weights, grad_HL1_HL2_weights,
                                                                      grad_HL2_output_weights,
                                                                      label_to_vector(
                                                                          batched_train_set_labels[batch][image],
                                                                          output_neurons), output, z0, a0, z1, a1, z2,
                                                                      a2)

                # add cost in list for last ploting
                cost += cost_MSE(output, batched_train_set_labels[batch][image])

                # print(predicted[0],"-----",train_set_labels[ (batch * batch_size) + image])

                # print(input_HL1_weights)

                grad_input_HL1_weights += tmpG1
                grad_HL1_HL2_weights += tmpG2
                grad_HL2_output_weights += tmpG3

                # update weights
                input_HL1_weights, HL1_HL2_weights, HL2_output_weights = update_weights(input_HL1_weights,
                                                                                        HL1_HL2_weights,
                                                                                        HL2_output_weights,
                                                                                        learning_rate,
                                                                                        batch_size,
                                                                                        grad_input_HL1_weights,
                                                                                        grad_HL1_HL2_weights,
                                                                                        grad_HL2_output_weights)
        costList.append(cost)

    plt.plot(list(range(len(costList))), np.array(costList), 'o')
    plt.plot(list(range(len(costList))), np.array(costList))
    plt.show()
    Test_Model.test_model(input_HL1_weights,HL1_HL2_weights,HL2_output_weights,test_set_features,test_set_labels)



if __name__ == "__main__":
    train_backpropagation200()
