import itertools
import math

from if_robustness.graph import Graph, Node


def test_graph():
    """Test if the sampled data corresponds to the conditionals from the graph"""

    r1 = Node(2, 'root 1')
    r2 = Node(2, 'root 2')
    c1 = Node(3, 'child 1')
    c1.add_parent(r1)
    c1.add_parent(r2)
    nodes = [r1, r2, c1]
    graph = Graph(nodes=nodes)
    graph.sample(100_000)

    observation_counts = graph.X.groupby(list(graph.X.columns)).size()
    observation_probabilities = observation_counts / observation_counts.groupby(['root 1', 'root 2']).transform('sum')

    for root1_value, root2_value, child_value in itertools.product(range(r1.cardinality), range(r2.cardinality),
                                                                   range(c1.cardinality)):
        expected = c1.get_probability(child_value, (root1_value, root2_value))
        result = observation_probabilities[root1_value, root2_value, child_value]
        assert math.isclose(expected, result, abs_tol=.015)
