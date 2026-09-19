-- The path artefact and the three dictionaries it joins to.
--
-- Everything here is read by people who will never run the engine, with ordinary
-- database operations and no decoding logic that exists only inside the decision
-- system (spec §5.4.1(b)).  There is exactly one non-obvious column in the whole
-- schema - route_digest - and one join that explains it.

-- ===========================================================================
-- THE FACT TABLE.  ~400 M rows per monthly cycle.  ~38 bytes/row compressed.
-- 400 M x 38 B = 15 GB, against the 120 GB budget of spec §5.4.1(a).
-- The margin is deliberate: the budget must hold when 60 campaigns becomes 90.
-- ===========================================================================
CREATE TABLE paths (
    cycle_id                INTEGER   NOT NULL,
    campaign_id             SMALLINT  NOT NULL,
    client_id               BIGINT    NOT NULL,
    tree_version            SMALLINT  NOT NULL,
    variant                 TINYINT   NOT NULL,   -- 0 champion, 1/2 challenger
    is_control              BOOLEAN   NOT NULL,   -- CONTROLS HAVE PATHS. §5.7 req 4
    overlay_stack_id        INTEGER   NOT NULL,   -- §5.3.4 req 3; NEVER null

    route_digest            BIGINT    NOT NULL,   -- uint64. THE PATH. 8 bytes.
    leaf_key                CHAR(18)  NOT NULL,
    route_depth             TINYINT   NOT NULL,

    -- The overlays-off twin.  Populated for every evaluation; the pair is what
    -- answers §9.5 ("what would we have done without the overlays?") from stored
    -- data with no re-run.  Equal to the adjusted columns for ~96% of rows, so it
    -- compresses to almost nothing.
    unadjusted_route_digest BIGINT    NOT NULL,
    unadjusted_leaf_key     CHAR(18)  NOT NULL,

    assignment_id           BIGINT             -- null where no assignment resulted
)
PARTITION BY (cycle_id, campaign_id)
CLUSTER BY (route_digest);

-- ===========================================================================
-- DICTIONARY 1: the route dictionary.  ~150 k rows per (campaign, tree version).
-- THIS is what serves "one row per node visited" without materialising 3 bn rows
-- (spec §13 Q3).  The explosion is against the dictionary, not the fact table.
-- ===========================================================================
CREATE TABLE route_node (
    campaign_id   SMALLINT NOT NULL,
    tree_version  SMALLINT NOT NULL,
    route_digest  BIGINT   NOT NULL,
    ordinal       TINYINT  NOT NULL,
    node_key      CHAR(18) NOT NULL,
    direction     BOOLEAN  NOT NULL,   -- the branch TAKEN, so a visited-and-failed
                                       -- node is present with direction = false
    edge_id       BIGINT   NOT NULL,
    next_key      CHAR(18) NOT NULL,
    PRIMARY KEY (campaign_id, tree_version, route_digest, ordinal)
);

-- ===========================================================================
-- DICTIONARY 2: node metadata.  Published at validation, retained indefinitely.
-- ===========================================================================
CREATE TABLE node_meta (
    campaign_id         SMALLINT NOT NULL,
    tree_version        SMALLINT NOT NULL,
    node_key            CHAR(18) NOT NULL,
    kind                VARCHAR(8)  NOT NULL,   -- internal | leaf
    level               TINYINT,
    parent_lineage      VARCHAR(512),
    author_label        VARCHAR(128),
    readable_condition  VARCHAR(1024),
    features            VARCHAR(1024),          -- ';'-separated feature ids
    on_true             CHAR(18),
    on_false            CHAR(18),
    leaf_outcome_code   TINYINT,
    offer_tier_code     TINYINT,
    amount_rule         VARCHAR(24),
    amount_cap_cents    BIGINT,
    channel_pref        VARCHAR(64),
    priority_weight     DOUBLE,
    reason_label        SMALLINT,
    PRIMARY KEY (campaign_id, tree_version, node_key)
);

-- ===========================================================================
-- DICTIONARY 3: the suppression bit map.  Same trick, same shape, deliberately.
-- ===========================================================================
CREATE TABLE suppression_bit (
    registry_version  VARCHAR(16) NOT NULL,
    bit_index         TINYINT     NOT NULL,
    suppression_code  CHAR(3)     NOT NULL,
    scope             VARCHAR(10) NOT NULL,   -- all | campaign | channel | client
    class             VARCHAR(12) NOT NULL,   -- absolute | measurement
    description       VARCHAR(256),
    PRIMARY KEY (registry_version, bit_index)
);


-- ===========================================================================
-- §5.4.2(1) NODE-LEVEL VOLUME.  One campaign-cycle, target < 90 seconds.
-- 7 M fact rows for one campaign-cycle partition, broadcast against a 150 k
-- dictionary.  Note the four grouping columns that are NOT node_key: a node
-- volume reported without the stack, version, variant and control split it was
-- produced under is not a comparable figure and must not be presentable as
-- though it were (spec §5.3.4(a), acceptance criterion 18).
-- ===========================================================================
SELECT  rn.node_key,
        p.tree_version, p.overlay_stack_id, p.variant, p.is_control,
        count(*)                                        AS entered,
        countIf(rn.direction)                           AS left_true,
        countIf(NOT rn.direction)                       AS left_false,
        countIf(rn.direction) / count(*)                AS true_rate
FROM    paths p
JOIN    route_node rn
        ON  rn.campaign_id  = p.campaign_id
        AND rn.tree_version = p.tree_version
        AND rn.route_digest = p.route_digest
WHERE   p.cycle_id = 2026091 AND p.campaign_id = 23
GROUP BY 1,2,3,4,5;


-- ===========================================================================
-- §5.3.4(a) DECOMPOSING A VOLUME MOVE, WITHOUT A RE-RUN.
-- Campaign 23 targeted 412 000 in August and 198 000 in September. Why?
-- Three causes produce the identical volume figure and the answer must say which.
--
--   overlay    measured DIRECTLY, from the dual columns: the rows whose
--              adjusted and unadjusted routes differ.
--   tree       measured from the IDENTITY MAP partition: volume on nodes the map
--              classifies `carried_forward` is comparable; volume on `changed`,
--              `added` and `removed` nodes is attributable to the version.
--   population the residual.  Named last, and only after the other two are known,
--              which is the difference between this and "fewer clients qualified".
-- ===========================================================================
WITH overlay_effect AS (
    SELECT count(*) AS clients_moved_by_overlay
    FROM   paths
    WHERE  cycle_id = 2026091 AND campaign_id = 23
      AND  route_digest <> unadjusted_route_digest
),
carried AS (   -- nodes the identity map says are genuinely the same test
    SELECT node_key FROM node_identity_map
    WHERE  campaign_id = 23 AND from_version = 11 AND to_version = 12
      AND  classification = 'carried_forward'
)
SELECT  'overlay'    AS cause, clients_moved_by_overlay AS clients FROM overlay_effect
UNION ALL
SELECT  'tree_version', sum(aug.entered - sep.entered)
FROM    node_volume aug JOIN node_volume sep USING (node_key)
WHERE   aug.node_key NOT IN (SELECT node_key FROM carried)
UNION ALL
SELECT  'population', /* residual: total move less the two above */ 0;


-- ===========================================================================
-- §5.4.2(2) DEAD BRANCHES.  1 340 of 9 200 live nodes had no traffic in 90 days.
-- The ones the validator proved unreachable are a DEFECT and were caught at
-- publication; these are the rest, and they are a signal to the campaign owner.
-- ===========================================================================
SELECT  nm.campaign_id, nm.tree_version, nm.node_key, nm.author_label
FROM    node_meta nm
LEFT JOIN (
        SELECT DISTINCT rn.campaign_id, rn.tree_version, rn.node_key
        FROM   paths p JOIN route_node rn
               ON rn.campaign_id = p.campaign_id
              AND rn.tree_version = p.tree_version
              AND rn.route_digest = p.route_digest
        WHERE  p.cycle_id IN (2026071, 2026081, 2026091)
) seen USING (campaign_id, tree_version, node_key)
WHERE   seen.node_key IS NULL AND nm.kind = 'internal';


-- ===========================================================================
-- §9.4 INTERNAL AUDIT, 18 MONTHS LATER.
-- "List every client excluded from campaign 17 in the March cycle, with the reason."
-- No re-run.  Suppressions are a bitmask exploded against the dictionary; the
-- arbitration ledger supplies the qualifying-but-not-contacted half.
-- ===========================================================================
SELECT  s.client_id, b.suppression_code, b.description, b.scope, 'suppression' AS source
FROM    suppression_facts s
JOIN    suppression_bit b
        ON  b.registry_version = s.registry_version
        AND ((s.supp_mask_global >> b.bit_index) & 1) = 1
WHERE   s.cycle_id = 2025031 AND s.campaign_id = 17
UNION ALL
SELECT  a.client_id, a.arbitration_reason_code, r.description, 'arbitration', 'arbitration'
FROM    arbitration_ledger a JOIN arbitration_reason r USING (arbitration_reason_code)
WHERE   a.cycle_id = 2025031 AND a.campaign_id = 17 AND a.contacted = false;
