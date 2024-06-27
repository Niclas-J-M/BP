import matplotlib.pyplot as plt
import networkx as nx

def draw_multiple_policy_networks():
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i, ax in enumerate(axs):
        G = nx.DiGraph()
        G.add_node('Input', pos=(0, 0))
        G.add_node(f'Hidden_{i}', pos=(1, 0))
        G.add_node(f'Output_{i}', pos=(2, 0))

        G.add_edge('Input', f'Hidden_{i}')
        G.add_edge(f'Hidden_{i}', f'Output_{i}')

        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, ax=ax)
        ax.set_title(f'Policy Network {i+1}')

    plt.show()

draw_multiple_policy_networks()


def draw_dynamic_policy_network():
    G = nx.DiGraph()
    G.add_node('Input', pos=(0, 0))
    G.add_node('Shared Hidden', pos=(1, 0))

    heads = ['Head 1', 'Head 2', 'Head 3']
    positions = [(2, 1), (2, 0.5), (2, 0)]

    for head, pos in zip(heads, positions):
        G.add_node(head, pos=pos)
        G.add_edge('Shared Hidden', head)

    G.add_edge('Input', 'Shared Hidden')

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')

    plt.title('Dynamic Policy Network with Shared Layers and Multiple Heads')
    plt.show()

draw_dynamic_policy_network()
