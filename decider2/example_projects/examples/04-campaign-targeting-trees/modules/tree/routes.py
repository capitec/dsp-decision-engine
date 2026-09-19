"""Route enumeration and the route dictionary — spec §5.4, and the answer to
§13 Q3 ("one row per evaluation, or one row per node visited?  Can both be
served from one recording?").

Both, from one recording, because the explosion happens against the **dictionary**
and not against the fact table.

    paths            400 M rows/cycle    one uint64 route_digest per evaluation
    route_node       ~150 k rows/version the dictionary, exploded to one row per
                                         (route, ordinal, node, direction)

"one row per node visited" at 3 billion rows is then an ordinary join the
analytics team writes in SQL, never materialised:

    SELECT rn.node_key, count(*)
    FROM   paths p
    JOIN   route_node rn USING (campaign_id, tree_version, route_digest)
    WHERE  p.cycle_id = 2026091 AND p.campaign_id = 23
    GROUP  BY rn.node_key;

One cycle's partition of `paths` for one campaign is ~7 M rows against a 150 k
dictionary — a broadcast join, comfortably inside the 90-second budget of §8.

Nothing here is decoding logic that exists only inside the decision system
(§5.4.1(b)).  The route digest is a uint64 column and the dictionary is a table.
"""

from __future__ import annotations

from pydantic import BaseModel

from .canonical import EdgeId, LeafKey, NodeKey, RouteDigest
from .document import TreeDocument


class RouteRow(BaseModel):
    campaign_id: int
    tree_version: int
    route_digest: int          # uint64, the join key
    ordinal: int               # 0-based position in the walk
    node_key: NodeKey
    direction: bool            # the branch TAKEN at this node
    edge_id: int
    next_key: str              # the node or leaf arrived at


class RouteDictionary(BaseModel):
    campaign_id: int
    tree_version: int
    rows: list[RouteRow]
    route_count: int
    leaf_of_route: dict[int, LeafKey]


def enumerate_routes(doc: TreeDocument) -> RouteDictionary:
    """Depth-first enumeration of every **feasible** root-to-leaf walk.

    Feasible, not merely graph-reachable: the same interval/set propagation
    validate.check_reachability uses, so a walk whose conjunction is UNSAT never
    enters the dictionary and never appears as a zero-volume row in the node
    report.  That distinction is what separates spec §5.4.2(2)'s "genuinely
    unreachable, should have been caught at publication" from "population drift,
    a signal to the campaign owner".

    THE DECISION §5.4.1(d) EXISTS TO FORCE INTO THE OPEN
    ----------------------------------------------------
    A route contains **every node visited, with the direction taken**, including
    the ones whose condition did not hold.  The spec's worked example prints the
    path as `1 -> 4 -> 6 -> 8 -> 10 -> leaf 912` and then renders node 2 anyway.
    We store `1(T) 2(F) 4(T) 6(T) 8(T) 10(T) -> l_cf2839a54e0b761d`.

    Storing the held-only subsequence would be smaller by about 15%, and would
    make the rendering a reconstruction — you would have to re-walk the tree to
    discover which node sat between 1 and 4.  That reconstruction is exactly the
    "decoding logic that exists only inside the decision system" §5.4.1(b)
    forbids, and it would silently break the day a tree version's shape changed
    under a stored path.  15% is not worth it.
    """
    pass  # DFS with interval propagation; fold edge ids into a digest per walk


def route_of(edges: list[EdgeId]) -> RouteDigest:
    """Present for symmetry with the kernel.  The kernel folds as it walks; this
    folds a list.  `assert_modes_agree` covers the pair (see DEMANDS #9)."""
    pass


def publish_node_meta(doc: TreeDocument) -> list[dict]:
    """The other half of §5.4.1(b): node metadata per (campaign, tree_version),
    carrying node key, level, parent lineage, the condition **in readable form**,
    the features it references, the direction taken for each outcome and — for
    leaves — every value the leaf carries.

    "Level" and "parent lineage" are properties of the *version*, not of the node,
    which is why they live here and not in the key.  A node's level can change
    between versions while its identity does not; the report says so, and that is
    correct rather than confusing: it is the same test, asked later.
    """
    pass  # flatten the document into warehouse rows
