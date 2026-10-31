"""BlitzRank (Agrawal, Nguyen and Kiela, ICML 2026, https://arxiv.org/abs/2602.05448), ported from the authors'
package (blitzrank 0.1.0, engine/algorithms/tournament_graph) with Jev as the judge.

The method: each round asks the judge about a group of k documents and adds the answers to one preference graph. Its
transitive closure says, for each document, how many are known to be better (in_reach) and worse (out_reach).
Documents that beat each other in a circle form one strongly connected component (SCC) and count as tied. A document
is final once its relation to every other document is known. Rounds go on until top_m documents are final. The next
round's group takes one representative from each of the first k unfinished SCCs, in order of in_reach, so every
round adds edges between documents whose order is still unknown.

Where this differs from the authors' code:
- Their judge orders the k documents (a listwise LLM call), giving edges between neighbours. Jev cannot order k
  documents, so a round is one Jev request asking every pair of the group (documents sent once). That gives a full
  tournament, possibly with cycles, which the graph handles as SCCs.
- A pair already answered (earlier in this rerank, or in the cache) is not asked again.
- If no round can add an edge any more, ranking stops with what is known instead of raising.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from reranker.ranking import Winner

# (group of positions, the pairs in it still to ask) -> each pair's winner
Ask = Callable[[list[int], list[tuple[int, int]]], Awaitable[dict[tuple[int, int], Winner]]]


@dataclass
class NodeInfo:
    node: int
    in_reach: int  # documents known to be better (outside its SCC)
    out_reach: int  # documents known to be worse (outside its SCC)
    known: int  # documents whose relation to it is known, its SCC's other members included
    scc: frozenset[int]


class TournamentGraph:
    """Directed graph on positions 0..n-1: an edge u -> v says u beat v."""

    def __init__(self, count: int):
        self.count = count
        self.beats = [set() for _ in range(count)]

    def add(self, winner: int, loser: int) -> None:
        self.beats[winner].add(loser)

    def nodes(self) -> list[NodeInfo]:
        reach = [self._reachable(node) for node in range(self.count)]  # strictly reachable from node
        infos = []
        for node in range(self.count):
            scc = frozenset({node} | {other for other in reach[node] if node in reach[other]})
            others_in_scc = len(scc) - 1
            out_reach = len(reach[node] - scc)
            in_reach = sum(1 for other in range(self.count) if other not in scc and node in reach[other])
            infos.append(NodeInfo(node, in_reach, out_reach, in_reach + out_reach + others_in_scc, scc))
        return infos

    def _reachable(self, start: int) -> set[int]:
        """Every document reachable from start through one or more edges (start itself only if it is on a cycle)."""
        seen: set[int] = set()
        stack = list(self.beats[start])
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(self.beats[node] - seen)
        return seen


@dataclass
class BlitzResult:
    top: list[tuple[int, int]]  # the top_m, best first: (position, how many documents are known to be better)
    rest: list[int]  # the others, unordered: BlitzRank does not rank past top_m
    certain: bool  # False when it stopped before top_m were final (max_rounds, or nothing left to ask)
    rounds: int
    asked: int  # pairs put to the judge
    cycles: int


async def blitz_sort(
    count: int, k: int, top_m: int, ask: Ask, known: dict[tuple[int, int], Winner], max_rounds: int = 50
) -> BlitzResult:
    """count documents at positions 0..count-1 (in the search's order, as far as the caller knows it). known: answers
    already at hand, (i, j) with i < j. ask is called once a round with the group and its unanswered pairs."""
    top_m = min(top_m, count)
    graph = TournamentGraph(count)
    answers = dict(known)
    for pair, winner in answers.items():
        _add(graph, pair, winner)

    rounds = asked = 0
    group = list(range(min(k, count)))
    final: list[int] = []
    infos = _sorted(graph.nodes())
    while len(final) < top_m and rounds < max_rounds:
        missing = [(i, j) for index, i in enumerate(group) for j in group[index + 1 :]]
        missing = [(min(i, j), max(i, j)) for i, j in missing]
        missing = [pair for pair in missing if pair not in answers]
        if missing:
            rounds += 1
            asked += len(missing)
            for pair, winner in (await ask(group, missing)).items():
                answers[pair] = winner
                _add(graph, pair, winner)
        infos = _sorted(graph.nodes())
        final, next_group = _progress(infos, count, k)
        if set(next_group) == set(group) or len(next_group) < 2:
            break  # nothing new left to ask: rank with what is known
        group = next_group

    certain = len(final) >= top_m
    if not certain:  # rank with what is known: the graph's order
        final = [info.node for info in infos]
    in_reach = {info.node: info.in_reach for info in infos}
    top = [(node, in_reach[node]) for node in final[:top_m]]
    chosen = {node for node, _ in top}
    rest = [node for node in range(count) if node not in chosen]
    return BlitzResult(top, rest, certain, rounds, asked, _cycles(graph))


def _add(graph: TournamentGraph, pair: tuple[int, int], winner: Winner) -> None:
    i, j = pair
    if winner is Winner.FIRST:
        graph.add(i, j)
    elif winner is Winner.SECOND:
        graph.add(j, i)
    else:  # a tie: each "beats" the other, so they share an SCC
        graph.add(i, j)
        graph.add(j, i)


def _sorted(infos: list[NodeInfo]) -> list[NodeInfo]:
    """The authors' default order (ascending out-reach strategy): fewest known better first, then fewest known worse,
    then the search's order."""
    return sorted(infos, key=lambda info: (info.in_reach, info.out_reach, info.node))


def _progress(infos: list[NodeInfo], count: int, k: int) -> tuple[list[int], list[int]]:
    """The final documents (in order), and the next group: one representative from each of the first k unfinished
    SCCs."""
    final, representatives, seen = [], [], set()
    for info in infos:
        if info.known >= count - 1:
            final.append(info.node)
            continue
        if info.scc not in seen:
            seen.add(info.scc)
            representatives.append(info.node)
            if len(representatives) >= k:
                break
    return final, representatives


def _cycles(graph: TournamentGraph) -> int:
    """Three documents that beat each other in a circle, from the direct answers (ties not counted)."""
    count = 0
    n = graph.count
    beats = graph.beats
    for a in range(n):
        for b in beats[a]:
            for c in beats[b]:
                if a in beats[c] and a not in beats[b] and b not in beats[c] and c not in beats[a]:
                    count += 1
    return count // 3
