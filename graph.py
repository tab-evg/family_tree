from _ast import List
from collections import deque
from copy import deepcopy, copy
from math import copysign
from platform import version
from typing import Iterable, Dict, Set
from typing import Tuple, Any


class DSU:
    def __init__(self, unions: Iterable[Tuple[Any]] = None, items: Iterable[Any] = None):
        self._parents = dict()
        self._ranks = dict()
        if items is not None:
            for i in items:
                self.get(i)
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
        result = Graph()
        for v, nb in self._vertexes.items():
            result._vertexes[v] = copy(nb)
        return result

    def concat(self, other: 'Graph') -> 'Graph':
        g = copy(self)
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
        if e[1] not in self._vertexes:
            self._vertexes[e[1]] = set()

    def remove_edge(self, e):
        if e[0] in self._vertexes:
            self._vertexes[e[0]].remove(e[1])

    def reverse_edge(self, e):
        if e[0] in self._vertexes:
            if e[1] in self._vertexes[e[0]]:
                self._vertexes[e[0]].remove(e[1])
                self.add_edge((e[1], e[0]))

    def is_vertex_in_graph(self, v) -> bool:
        return v in self._vertexes

    def is_edge_in_graph(self, e) -> bool:
        return e[0] in self._vertexes and e[1] in self._vertexes[e[0]]

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


def spanning_tree_edges(g: Graph):
    '''
    Покрытие связного графа остовным деревом
    :return: список ребер остовного дерева
    '''
    # создаем ненаправленный граф из направленного
    gundir = g.get_reversed().concat(g)
    dfs_res = gundir.dfs()
    edges = set()
    for e in dfs_res.tree_edges:
        if g.is_edge_in_graph(e):
            edges.add(e)
        else:
            edges.add((e[1], e[0]))
    return edges


def tight_tree(g: Graph, ranks: Dict[Any, int]) -> Graph:
    delta = 1
    dsu = DSU()
    for e in g.iter_edges():
        if ranks[e[1]] - ranks[e[0]] == delta:
            # tight edge
            dsu.union(*e)
    max_tree_v = max(dsu.iter_unions(), key=lambda x: len(x))
    tight_tree = Graph()
    for v in max_tree_v:
        for u in g.iter_neighbors(v):
            if u in max_tree_v and ranks[u] - ranks[v] == delta:
                tight_tree.add_edge((v, u))

    return Graph(spanning_tree_edges(tight_tree))


def slack(edge, delta, ranks):
    return ranks[edge[1]] - ranks[edge[0]] - delta


def get_weight(edge: Tuple[Any]) -> int:
    return 1


def init_cutvalues(tree: Graph, g: Graph):
    res = dict()
    tree_edges = set(tree.iter_edges())
    for e in tree.iter_edges():
        dsu = DSU(tree_edges - {e}, tree.iter_vertices())
        # print(f'{e}: {[x for x in dsu.iter_unions()]}')
        tail_comp = dsu.get(e[0])
        head_comp = dsu.get(e[1])
        cut = 0
        for cur_e in g.iter_edges():
            tail = dsu.get(cur_e[0])
            head = dsu.get(cur_e[1])
            if tail == tail_comp and head == head_comp:
                cut += get_weight(cur_e)
            elif tail == head_comp and head == tail_comp:
                cut -= get_weight(cur_e)
        res[e] = cut
    return res


def feasible_tree(g: Graph) -> Dict[Any, int]:
    delta = 1
    ranks = init_rank(g)
    print(f'init ranks={ranks}')
    tree = tight_tree(g, ranks)
    min_slack_edge = None
    while tree.vertex_count() < g.vertex_count():
        print(tree)
        for v in g.iter_vertices():
            for u in g.iter_neighbors(v):
                if tree.is_vertex_in_graph(u) != tree.is_vertex_in_graph(v):
                    if min_slack_edge is None or slack((v, u), delta, ranks) < slack(min_slack_edge, delta, ranks):
                        min_slack_edge = (v, u)

        d = slack(min_slack_edge, delta, ranks)
        if g.is_vertex_in_graph(min_slack_edge[1]):
            d = -d
        for v in tree.iter_vertices():
            ranks[v] += d
        print(f'ranks={ranks}')
        tree = tight_tree(g, ranks)
    print(f'tree={tree}')
    cutvals = init_cutvalues(tree, g)
    return tree, cutvals, ranks


def find_key_by_val(d, func):
    for k, v in d.items():
        if func(v):
            return k


def enter_edge(tree: Graph, g: Graph, edge: Tuple[Any], ranks: Dict[Any, int]) -> Tuple[Any]:
    delta = 1
    res = None
    res_sl = None
    tree_edges = set(tree.iter_edges())
    dsu = DSU(tree_edges - {edge}, tree.iter_vertices())
    tail_comp = dsu.get(edge[0])
    head_comp = dsu.get(edge[1])

    for cur_e in g.iter_edges():
        tail = dsu.get(cur_e[0])
        head = dsu.get(cur_e[1])
        if tail == head_comp and head == tail_comp:
            sl = slack(cur_e, delta, ranks)
            if res is None or sl < res_sl:
                res = cur_e
                res_sl = sl
    return res


def calculate_rank(g: Graph):
    feas_tree, cutvalues, ranks = feasible_tree(g)
    print(f'feasible_tree={feas_tree}')
    print(f'cutvalues={cutvalues}')
    print(f'feas ranks={ranks}')
    while True:
        e = find_key_by_val(cutvalues, lambda x: x < 0)
        if e is None:
            break
        f = enter_edge(feas_tree, g, e, ranks)
        feas_tree.remove_edge(e)
        feas_tree.add_edge(f)
        print(f'feas={feas_tree}')
        cutvalues = init_cutvalues(feas_tree, g)

    rmin = min(ranks.values())
    ranks = {v: (r - rmin) for v, r in ranks.items()}
    return ranks


class VirtualNode:
    def __init__(self, edge, index):
        self._edge = edge
        self._index = index

    def __repr__(self):
        return f'[V{self._edge}: {self._index}]'

    @property
    def size(self) -> Tuple[int]:
        return 10, 10


class TestNode:
    def __init__(self, text, size=(10, 10)):
        self._text = text
        self._size = size

    def __repr__(self):
        return f'{self._text}'

    @property
    def size(self) -> Tuple[int]:
        return self._size


def add_virtual_nodes(g: Graph, ranks: Dict[Any, int]):
    long_edges = []
    for e in g.iter_edges():
        dr = ranks[e[1]] - ranks[e[0]]
        if abs(dr) > 1:
            long_edges.append((e, dr))

    virt_nodes = []
    for e, dr in long_edges:
        vl = list()
        prev = e[0]
        for i in range(1, abs(dr)):
            n = VirtualNode(e, i)
            g.add_edge((prev, n))
            vl.append(n)
            ranks[n] = ranks[e[0]] + (i if dr > 0 else -i)
            prev = n
        g.add_edge((prev, e[1]))
        virt_nodes.append(vl)
    return virt_nodes


def ordering(g: Graph, ranks: Dict[Any, int]) -> Dict:
    best_order = dict()
    for v, r in ranks.items():
        l = best_order.get(r, list())
        l.append(v)
        best_order[r] = l
    return best_order


def xcoordinate(g: Graph, ranks: Dict[Any, int], order: Dict, nodesep=10) -> Dict:
    ## init_xcoord()
    xcoords = dict()

    for nodes in order.values():
        x = 0
        for v in nodes:
            xcoords[v] = x
            x += v.size[0] + nodesep
    return xcoords


def ycoordinate(g: Graph, ranks: Dict[Any, int], order: Dict, ranksep=10) -> Dict:
    ycoords = dict()
    y = 0
    for nodes in order.values():
        max_height = 0
        for v in nodes:
            ycoords[v] = y
            max_height = max(max_height, v.size[1])
        y += max_height + ranksep
    return ycoords


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
    ta = TestNode('a')
    tb = TestNode('b')
    tc = TestNode('c')
    td = TestNode('d')
    te = TestNode('e')
    tf = TestNode('f')
    tg = TestNode('g')
    th = TestNode('h')

    gt = Graph(
        [(ta, tb),
         (tb, tc),
         (tc, td),
         (td, th),
         (ta, tf),
         (tf, tg),
         (tg, th),
         (ta, te),
         (te, tg)])
    make_acyclic(gt)
    print(f'acyclic={gt}')
    ranks = calculate_rank(gt)
    print(f'ranks={ranks}')
    virtual_nodes = add_virtual_nodes(gt, ranks)
    print(f'vgraph={gt}')
    print(f'virtual_nodes={virtual_nodes}')
    order = ordering(gt, ranks)
    print(f'order={order}')
    xcoord = xcoordinate(gt, ranks, order)
    print(f'xcoord={xcoord}')
    ycoord = ycoordinate(gt, ranks, order)
    print(f'xcoord={ycoord}')
