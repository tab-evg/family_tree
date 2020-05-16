from collections import deque
from copy import deepcopy
from platform import version
from typing import Iterable, Dict, Set
from typing import Tuple, Any


class DSU:
    def __init__(self, unions: Tuple[Any] = None):
        self._parents = dict()
        self._ranks = dict()
        if unions is not None:
            for u in unions:
                self.union(*u)

    def get(self, x):
        if x in self._parents:
            if self._parents[x] != x:
                self._parents[x] = self.get(self._parents[x])
            return self._parents[x]
        else:
            self._parents[x] = x
            self._ranks[x] = 0
            return x

    def union(self, x, y):
        x = self.get(x)
        y = self.get(y)
        if x == y:
            return
        if self._ranks[x] == self._ranks[y]:
            self._ranks[x] += 1
        if self._ranks[x] < self._ranks[y]:
            self._parents[x] = y
        else:
            self._parents[y] = x

    def iter_unions(self) -> Iterable[Set[Any]]:
        unions = dict()
        for x, parent in self._parents.items():
            if parent != x:
                parent = self.get(parent)
            u = unions.get(parent, set())
            u.add(x)
            unions[parent] = u

        for u in unions.values():
            yield u


class DFSResult:
    def __init__(self):
        self.parent = {}
        self.start_time = {}
        self.finish_time = {}
        self.tree_edges = set()
        self.back_edges = set()
        self.forward_edges = set()
        self.cross_edges = set()
        self.order = []
        self.t = 0

    def __str__(self):
        res = f'parent ={self.parent}\n' + \
              f'start_time ={self.start_time}\n' + \
              f'finish_time ={self.finish_time}\n' + \
              f'tree_edges ={self.tree_edges}\n' + \
              f'back_edges ={self.back_edges}\n' + \
              f'forward_edges ={self.forward_edges}\n' + \
              f'cross_edges ={self.cross_edges}\n' + \
              f'order ={self.order}\n' + \
              f't ={self.t}\n'
        return res


class Graph:
    def __init__(self, edges: Iterable[Tuple[Any]] = None, vertexes: Iterable[Any] = None):
        self._vertexes = dict()
        if vertexes is not None:
            for v in vertexes:
                self._vertexes[v] = set()

        if edges is not None:
            for e in edges:
                neighbours = self._vertexes.get(e[0], set())
                neighbours.add(e[1])
                self._vertexes[e[0]] = neighbours
                if e[1] not in self._vertexes:
                    self._vertexes[e[1]] = set()

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def concat(self, other: 'Graph') -> 'Graph':
        g = deepcopy(self)
        for v, neighbours in other._vertexes.items():
            this_neighbours = self._vertexes.get(v, set())
            this_neighbours.update(neighbours)
            g._vertexes[v] = this_neighbours
        return g

    def __repr__(self):
        return str(self._vertexes)

    def vertex_count(self):
        return len(self._vertexes)

    def add_vertex(self, v):
        if v not in self._vertexes:
            self._vertexes[v] = set()

    def add_edge(self, e):
        neighbours = self._vertexes.get(e[0], set())
        neighbours.add(e[1])
        self._vertexes[e[0]] = neighbours

    def remove_edge(self, e):
        if e[0] in self._vertexes:
            self._vertexes[e[0]].remove(e[1])

    def reverse_edge(self, e):
        if e[0] in self._vertexes:
            if e[1] in self._vertexes[e[0]]:
                self._vertexes[e[0]].remove(e[1])
                self.add_edge((e[1], e[0]))

    def get_reversed(self):
        edges = []
        for v in self.iter_vertices():
            for u in self.iter_neighbors(v):
                edges.append((u, v))
        return Graph(edges)

    def iter_vertices(self):
        for v in self._vertexes.keys():
            yield v

    def iter_edges(self):
        for v, neighbors in self._vertexes.items():
            for nb in neighbors:
                yield v, nb

    def iter_neighbors(self, v):
        for u in self._vertexes.get(v, set()):
            yield u

    def get_neighbors_count(self, v):
        return len(self._vertexes.get(v, set()))

    def dfs(self):

        results = DFSResult()

        for vertex in self.iter_vertices():
            if vertex not in results.parent:
                self.__dfs_visit(vertex, results)
        return results

    def __dfs_visit(self, v, results: DFSResult, parent=None, callback=None):

        results.parent[v] = parent

        results.t += 1
        results.start_time[v] = results.t
        if callback is not None:
            callback(v)
        if parent:
            results.tree_edges.add((parent, v))

        for n in self.iter_neighbors(v):
            if n not in results.parent:  # n is not visited.
                self.__dfs_visit(n, results, v, callback)
            elif n not in results.finish_time:
                results.back_edges.add((v, n))
            elif results.start_time[v] < results.start_time[n]:
                results.forward_edges.add((v, n))
            else:
                results.cross_edges.add((v, n))

        results.t += 1
        results.finish_time[v] = results.t
        results.order.append(v)

    def get_scc_map(self):
        forward_dfs = self.dfs()
        rev_g = self.get_reversed()

        components = dict()
        scc_num = 0
        back_dfs = DFSResult()

        def add_index(v):
            components[v] = scc_num

        for vertex in reversed(forward_dfs.order):
            if vertex not in back_dfs.parent:
                rev_g.__dfs_visit(vertex, back_dfs, None, add_index)
                scc_num += 1
        return components

    def get_condensation(self):
        сomponents = self.get_scc_map()
        g_condensat = Graph()
        g_scc = dict()
        for v, scc_num in сomponents.items():
            if scc_num not in g_scc:
                g_scc[scc_num] = Graph()
            g_scc[scc_num].add_vertex(v)
            for u in self.iter_neighbors(v):
                if scc_num == сomponents[u]:
                    g_scc[scc_num].add_edge((v, u))
                else:
                    g_condensat.add_edge((scc_num, сomponents[u]))
        return g_condensat, g_scc


def make_acyclic(g: Graph):
    # TODO нужно реализовать умную эвристику
    dfs_res = g.dfs()
    for e in dfs_res.back_edges:
        g.reverse_edge(e)


def init_rank(g: Graph) -> Dict[Any, int]:
    delta = 1
    ranks = {v: 0 for v in g.iter_vertices()}
    rev_g = g.get_reversed()
    in_edges = g.get_reversed()
    dfs_res = g.dfs()
    print(dfs_res)
    q = deque()
    for v in reversed(dfs_res.order):
        if in_edges.get_neighbors_count(v) == 0:
            q.append(v)
    while len(q) > 0:
        v = q.popleft()
        # вычисляем ранг
        ranks[v] = max(map(ranks.get, rev_g.iter_neighbors(v)), default=0) + delta
        # for x in rev_g.iter_neighbors(v):
        #     print(x, end=' ')
        # print(f':{v}<-{ranks[v]}')
        for u in g.iter_neighbors(v):
            in_edges.remove_edge((u, v))
            if in_edges.get_neighbors_count(u) == 0:
                q.append(u)
    return ranks


def tight_tree(g: Graph, ranks: Dict[Any, int]) -> Set[Any]:
    # !!!TODO DSF SPANNING TREE
    delta = 1
    dsu = DSU()
    for e in g.iter_edges():
        if ranks[e[1]] - ranks[e[0]] == delta:
            # tight edge
            dsu.union(*e)
    return max(dsu.iter_unions(), key=lambda x: len(x))


if __name__ == '__main__':
    # g1 = Graph([(1, 2), (2, 3), (3, 4)])
    # print(g1.concat(g1.get_reversed()))
    # dsu = DSU([(1, 2), (2, 3), (4, 5), (5, 8), (10, 11), (15, 15)])
    # for u in dsu.iter_unions():
    #     print(u)

    # gt = Graph([(1, 2), (1, 3), (3, 4), (4, 5), (4, 3), (5, 3), (1, 4), (3, 1)])
    # print(str(gt.dfs()))
    # print(gt.get_scc_map())
    # print(gt.get_condensation())
    # gt.reverse_edge((4, 3))
    # print(str(gt.dfs()))
    # print(str(gt.get_reversed().dfs()))
    gt = Graph(
        [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'h'), ('a', 'f'), ('f', 'g'), ('g', 'h'), ('a', 'e'), ('e', 'g')])
    make_acyclic(gt)
    print(f'acyclic={gt}')
    ranks = init_rank(gt)
    print(f'ranks={ranks}')
    tree = tight_tree(gt, ranks)
    print(tree)
