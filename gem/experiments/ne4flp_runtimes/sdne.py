from gem.embedding.sdne import SDNE
import networkx as nx
from timeit import Timer

sizes = [200, 500, 1000, 2000, 3000, 5000, 10000, 15000, 20000, 35000, 50000]
_DENS = 1e-3


for s in sizes:
    G = nx.fast_gnp_random_graph(s, _DENS, directed=True)
    sdne_ = SDNE(d=128,
                 beta=5,
                 alpha=1e-5,
                 nu1=1e-6,
                 nu2=1e-6,
                 K=3,
                 n_units=[50, 15],
                 rho=0.3,
                 n_iter=10,
                 xeta=0.01,
                 n_batch=500)
    t = Timer('sdne_.learn_embedding(G)', setup='from __main__ import sdne_, G')
    n_runs = 3 if s <= 5000 else 1
    exec_times = t.repeat(n_runs, 1)
    print(f'{s}: {exec_times}')

    with open('sdne_times.txt', 'a') as f:
        f.write(f'{s}: {exec_times}\n')
