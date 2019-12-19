import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


class Node:
    """A node is a categorical value in range 0 to cardinality, depending on its parents

    You can sample observations from a node based on its conditional probabilities. To this end you first need to
    1. set the parents using 'add_parents'
    2. set the conditionals using 'set_conditionals'
    3. sample values for its parents
    """

    def __init__(self, cardinality, name, debug=False):
        self.cardinality = cardinality
        self.conditionals = None
        self.debug = debug
        self.feature_value = None
        self.name = name
        self.parents = []

    def add_parent(self, parent):
        self.parents.append(parent)

    def set_conditionals(self):
        """Set random conditional probabilities"""

        self.conditionals = np.random.rand(*[p.cardinality for p in self.parents], self.cardinality)
        # divide probability based on random numbers
        for i in itertools.product(*[range(p.cardinality) for p in self.parents]):
            self.conditionals[i] = pd.Series(self.conditionals[i]).pipe(lambda x: x / sum(x)).cumsum()

        if self.debug:
            print(self.name, 'Conditionals:', self.conditionals)

    def sample_value(self):
        """Sample a value for this node based on the conditionals and its parent's values"""

        parent_values = tuple(p.feature_value for p in self.parents)
        if None in parent_values:
            print(self.name, 'Error: not all parent values are set.')

        if not self.parents:
            probability_thresholds = self.conditionals
        else:
            probability_thresholds = self.conditionals[parent_values]

        random_value = random.random()
        for feature_value, probability_threshold in enumerate(probability_thresholds):
            if random_value <= probability_threshold:
                self.feature_value = feature_value
                break
        return self.feature_value

    def get_probability(self, own_value, parent_values: tuple = ()):
        """Get the expected probability for an observation"""

        if own_value == 0:
            return self.conditionals[parent_values][own_value]
        else:
            return self.conditionals[parent_values][own_value] - self.conditionals[parent_values][own_value-1]

    def __repr__(self):
        return f'{self.name} ({self.cardinality})'

    def __str__(self):
        return self.__repr__()


class Chain:
    """A single chain of connected nodes"""

    def __init__(self, length, cardinality_lower=2, cardinality_upper=10):
        self.length = length
        self.nodes = [Node(random.randint(cardinality_lower, cardinality_upper), 'chain 0')]
        for i in range(length - 1):
            node = Node(random.randint(cardinality_lower, cardinality_upper), f'chain {i + 1}')
            node.add_parent(self.nodes[i])
            self.nodes.append(node)

    def __repr__(self):
        return str([n for n in self.nodes])


class Graph:
    """
    A set of one or more connected acyclic components from which data can be generated and anomalies identified.

    The anomalies are only different on the nodes and not on the chain_nodes and external_nodes.
    The sampled feature values will be saved to X and the labels indicating the anomalies to y.

    Note: the order of the nodes is important; all parents of a node need to be ordered before itself (no cycles)

    Usage:
    parent = Node(3, 'parent with cardinality 3')
    child = Node(5, 'child with cardinality 5')
    child.add_parent(parent)
    graph = Graph(nodes=[parent, child])
    graph.sample()
    graph.label(nr_features_to_change=2)
    graph.score()
    """

    def __init__(self, nodes, contamination=.005, nr_externals=0, nr_chains=0, chain_length=10,
                 debug=False):
        self.contamination = contamination
        self.debug = debug
        self.number_observations = None
        self.X = None
        self.y = None

        self.chain_nodes = []
        for _ in range(nr_chains):
            for n in Chain(length=chain_length).nodes:
                self.chain_nodes.append(n)
        self.external_nodes = [Node(random.randint(2, 10), f'external {i}') for i in range(nr_externals)]
        self.nodes = nodes
        self.all_nodes = self.nodes + self.external_nodes + self.chain_nodes
        for n in self.all_nodes:
            n.set_conditionals()

    def sample(self):
        self.X = pd.DataFrame(data=[[n.sample_value() for n in self.all_nodes]
                                    for _ in range(self.number_observations)],
                              columns=[n.name for n in self.all_nodes])

    def label(self, number_observations=10_000, nr_features_to_change=0, noise=10):
        """Put anomalies in the data and label the data accordingly"""

        if nr_features_to_change > len(self.nodes):
            print(f'Error: not possible to change {nr_features_to_change} out of {len(self.nodes)} features')
            return

        self.number_observations = number_observations
        self.sample()
        self.y = np.zeros(self.number_observations)
        for _ in range(int(self.number_observations * self.contamination)):
            observation_id = random.randint(0, self.number_observations - 1)
            self.y[observation_id] = 1
            feature_set = set(range(len(self.nodes)))
            for _ in range(nr_features_to_change):
                random_feature_id = random.choice(tuple(feature_set))
                feature_set -= {random_feature_id}
                self.X.iloc[observation_id, random_feature_id] = (
                    self.nodes[random_feature_id].cardinality + random.randint(0, noise))

    def score(self, model=None):
        """Train an Isolation Forest to identify the anomalies and return the result"""

        if self.y is None:
            print('Error: first label the data.')
            return

        if model is None:
            model = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1)

        model.fit(self.X)
        predictions = model.predict(self.X)
        y_pred = pd.Series(data=[1 if x == -1 else 0 for x in predictions])
        y_score = pd.Series(data=model.decision_function(self.X)).multiply(-1)
        fpr, tpr, thresholds = roc_curve(self.y, y_score)
        roc_auc = auc(fpr, tpr)

        if self.debug:
            print('Nr. of features: ', self.X.shape[1])
            print('Labels:\n', pd.Series(self.y).value_counts())
            print('\nConfusion matrix\n', confusion_matrix(self.y, y_pred))
            # print('\n ROC AUC score', round(roc_auc_score(self.y, y_score), 2))
            # print('\nClassification report\n', classification_report(self.y, y_pred))
            self.plot_roc(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        return roc_auc


    @staticmethod
    def plot_roc(fpr, tpr, roc_auc):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
