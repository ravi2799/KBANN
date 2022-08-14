import os
import tensorflow as tf
import numpy as np
from sklearn import mixture
from Dataset import load_data
from rule_to_network import load_rules,rewrite_rules
from rule_to_network import rule_to_network

DEFAULT_WEIGHT = 4.0

def cluster_weights(links, threshold):
    weights = np.transpose(np.array([links]))

    # Clustering links
    n = len(links)
    MIN_NUM_SAMPLES = 2
    if n > MIN_NUM_SAMPLES:
        # Fit a mixture of Gaussians with EM
        lowest_bic = np.infty
        bic = []
        for n_components in range(2, n):
            gmm = mixture.GMM(n_components=n_components, covariance_type='full')
            gmm.fit(weights)
            # Bayesian information criterion
            bic.append(gmm.bic(weights))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

        # Average weights
        ids = best_gmm.predict(weights)
        unique_ids = list(set(ids))

        for i in unique_ids:
            indices = ids == i
            average_weight = np.sum(links[indices]) / len(links[indices])
            links[indices] = average_weight

        return links, ids
    elif n == 2:
        return links, np.array([0, 1])
    else:
        return links, np.zeros(len(links))


def preprocess_data(dataset, feature_names, layers):

    last_layer = []; X = []; i = 1
    for layer in layers:
        indices = []
        if i == 1:
            # input layer
            indices = [feature_names.index(unit) for unit in layer]
            X.append(dataset[:, indices])

        elif i % 2 != 0 and len(last_layer) > 0:
            # hidden and output layer
            hidden_input = [unit for unit in layer if unit not in last_layer]
            indices = [feature_names.index(unit) for unit in hidden_input]
            x = dataset[:, indices]
            n = len(x)
            m = len(x[0])
            X.append(x + 0.00001 * np.random.rand(n, m))
        else:
            last_layer = layer
        i += 1

    return X

def eliminate_weights(weights, biases):
    cluster_ids = []
    for i in range(len(weights)):
        cluster = []
        for j in range(weights[i].shape[1]):
            b = biases[i][0, j]
            (_w, ids) = cluster_weights(weights[i][:, j], b)
            weights[i][:, j] = list(_w)
            cluster.append(ids)
        cluster_ids.append(cluster)

    return weights, biases, cluster_ids

def network_to_rule(weights, biases, cluster_indices, layers):
    
    rules = []
    layers = np.array(layers)
    weight_range = range(0, len(weights))
    layer_range = range(0, len(layers), 2)
    for i, l in zip(weight_range, layer_range):
        current_layer = np.array(layers[l])
        next_layer = layers[l + 1]
        for j in range(weights[i].shape[1]):
            b = biases[i][0, j]
            w = weights[i][:, j]
            head = next_layer[j]
            indices = cluster_indices[i][j]
            unique_ids = list(set(indices))
            body = ''
            for id in unique_ids:
                if body != '':
                    body += ' + '
                matched_indices = indices == id
                antecedents = current_layer[matched_indices]
                threshold = w[matched_indices]
                body += str(threshold[0]) + ' * nt(' + ','.join(antecedents) + ')'
            new_rule = head + ' :- ' + str(b) + ' < ' + body
            rules.append(new_rule)

            print(head + ' = 0')
            print('if ' + str(b) + ' < ' + body + ':')
            print('\t' + head + ' = 1')
    return rules

def add_input_units(weights, layers, feature_names):

    additional_units = feature_names.copy()

    for layer in layers:
        for unit in layer:
            if unit in feature_names:
                additional_units.remove(unit)

    w = weights[0]
    zeros = np.zeros((len(additional_units), w.shape[1]))
    weights[0] = np.row_stack([w, zeros])
    layers[0] += additional_units
    return weights, layers

def add_hidden_units(weights, biases, layers):
    """Add units to hidden layers
    """
    w1 = weights[0]
    w2 = weights[1]
    zeros1 = np.zeros((w1.shape[0], 3))
    weights[0] = np.column_stack([w1, zeros1])
    zeros2 = np.zeros((3, 1))

    weights[1] = np.row_stack([w2, zeros2])
    b = biases[0]
    biases[0] = np.column_stack([b, np.zeros((1, 3))])

    layers[1].insert(len(layers[1]), 'head1')
    layers[1].insert(len(layers[1]), 'head2')
    layers[1].insert(len(layers[1]), 'head3')

    layers[2].insert(len(layers[2]), 'head1')
    layers[2].insert(len(layers[2]), 'head2')
    layers[2].insert(len(layers[2]), 'head3')

    return weights, biases, layers

def simplify_rules(rules):

    return rules

def save(rules, filepath):
    with open(filepath, 'wb') as f:
        for row in rules:
            f.write(repr(str(row)) + '\n')

class KBANN:
    def __init__(self, weights, biases, X, y, learning_rate):

        self.weights = self.set_weights(weights)
        self.biases = self.set_biases(biases)
        self.num_layers = len(weights)
        self.input_data = []
        self.input_mask = []
        self.learning_rate = learning_rate
        for x in X:
            self.input_data.append(tf.constant(x, dtype=tf.float32))
            if x.shape[1] > 0:
                self.input_mask.append(True)
                print('masked')
            else:
                self.input_mask.append(False)

        self.targets = tf.constant(y, dtype=tf.float32)

    def propagate_forward(self):

        activations = [tf.sigmoid(tf.matmul(self.input_data[0], self.weights[0]) - self.biases[0])]
        for i in range(1, self.num_layers):
            if self.input_mask[i]:
                input_tensor = tf.concat(1, [activations[-1], self.input_data[i]])
            else:
                input_tensor = activations[-1]
            activation = tf.sigmoid(tf.matmul(input_tensor, self.weights[i]) - self.biases[i])
            activations.append(activation)
        return activations[-1]

    def propagate_forward_with_dropout(self):
        """Implements the forward propagation"""

        activations = [tf.sigmoid(tf.matmul(self.input_data[0], self.weights[0]) - self.biases[0])]

        for i in range(1, self.num_layers):
            if self.input_mask[i]:
                input_tensor = tf.concat(1, [activations[-1], self.input_data[i]])
            else:
                input_tensor = activations[-1]
            dropout_layer = tf.nn.dropout(input_tensor, 0.8)
            activation = tf.sigmoid(tf.matmul(dropout_layer, self.weights[i]) - self.biases[i])
            activations.append(activation)
        return activations[-1]

    def compute_loss(self, logits):
        return tf.reduce_mean(-self.targets * tf.log(logits) - (1.0 - self.targets) * tf.log(1.0 - logits))

    def train(self, cost):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        return optimizer.minimize(cost)

def display(arrays):
    for array in arrays:
        print(array)

if __name__ == "__main__":

    CURRENT_DIRECTOR = os.getcwd()

    # Initial parameters
    training_epochs = 2000000
    display_step = 10000
    learning_rate = 0.1

    # Load training data
    data_file_path = CURRENT_DIRECTOR + '/Datasets/trial_data.txt'
    X, y, feature_names = Dataset.load_data(data_file_path)

    # Translate rules to a network
    rule_file_path = CURRENT_DIRECTOR + '/Datasets/trial_data_rules.txt.txt'
    ruleset = rule_to_network.load_rules(rule_file_path)
    ruleset = rule_to_network.rewrite_rules(ruleset)
    weights, biases, layers = rule_to_network.rule_to_network(ruleset)
    display(layers)
    print('---------------------')

    weights, layers = add_input_units(weights, layers, ['complete_course', 'freshman', 'sent_application', 'high_gpa'])

    weights, biases, layers = add_hidden_units(weights, biases, layers)
    display(layers)

    # Pre-process input data
    X = preprocess_data(X, feature_names, layers)

    print('Parameters 0:')
    display(weights)
    display(biases)

    # Construct a training model
    model = KBANN(weights, biases, X, y, learning_rate) # tensorflow model

    # Launch the graph
    with tf.Session() as session:

        # Initialize all variables
        session.run(tf.initialize_all_variables())

        logits = model.propagate_forward_with_dropout()
        loss = model.compute_loss(logits)
        train_op = model.train(loss)
        weights = session.run(model.weights)
        biases = session.run(model.biases)

        # Refine rules
        for epoch in range(training_epochs):
            session.run(train_op)
            if epoch % display_step == 0:
                loss_value = session.run(loss)
                print('Epoch %d: Loss = %.9f' % (epoch / display_step, loss_value))

        weights = session.run(model.weights)
        biases = session.run(model.biases)


        weights, biases, cluster_indices = eliminate_weights(weights, biases)
        print('Parameters 3:')
        display(weights)
        display(biases)

        model.set_weights(weights, fixed=True)
        model.set_biases(biases)
        logits = model.propagate_forward()
        loss = model.compute_loss(logits)
        train_op = model.train(loss)

        for epoch in range(1000000):
            _, loss_value = session.run([train_op, loss])
            if epoch % display_step == 0:
                print('Epoch %d: Loss = %.9f' % (epoch / display_step, loss_value))
        biases = session.run(model.biases)

        # Translate network to rules
        ruleset = network_to_rule(weights, biases, cluster_indices, layers)
        print('Parameters 4:')
        display(weights)
        display(biases)