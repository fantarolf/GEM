from gem.embedding.lap import LaplacianEigenmaps
import networkx as nx
from timeit import Timer

sizes = [200, 500, 1000, 2000, 3000, 5000, 10000, 15000, 20000, 35000, 50000]
_DENS = 1e-3


for s in sizes:
    G = nx.gnp_random_graph(s, _DENS, directed=True)
    l_eig = LaplacianEigenmaps(d=128)
    t = Timer('l_eig.learn_embedding(G)', setup='from __main__ import l_eig, G')
    n_runs = 3 if s <= 5000 else 1
    exec_times = t.repeat(n_runs, 1)
    print(f'{s}: {exec_times}')

    with open('l_eig_times.txt', 'a') as f:
        f.write(f'{s}: {exec_times}\n')
