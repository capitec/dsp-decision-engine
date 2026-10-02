// Mock stand-ins for the project-owned `experiments/<slug>/experiment.json` assets — just enough
// to drive the sidebar's Experiments pane (name + search) and open the summary.

import type { ExperimentDef } from "../../src/model/experiment";

export interface MockExperiment {
  slug: string;
  name: string;
  definition: ExperimentDef;
  /** Has a `result.json` already, so its summary offers "Open last report". */
  hasResult?: boolean;
}

export const EXPERIMENTS: MockExperiment[] = [
  {
    slug: "affordability-verdict-drift",
    name: "affordability-verdict-drift",
    definition: {
      version: "2",
      name: "affordability-verdict-drift",
      description: "Does raising the affordability ratio limit change who is approved?",
      flow: { entry: "loan:pipeline" },
      scenarios: [{ name: "limit_0.5", params: { affordability: { affordable: { limit: 0.5 } } } }],
      metrics: [{ type: "divergence", column: "offer" }],
      plots: [{ type: "changes", metric: "divergence" }],
    },
  },
  {
    slug: "term-cap-sensitivity",
    name: "term-cap-sensitivity",
    definition: {
      version: "2",
      name: "term-cap-sensitivity",
      description: "How far does the term cap move offers when the income cap changes?",
      flow: { entry: "loan:pipeline" },
      scenarios: [{ name: "cap_24", params: { term: { cap_by_income: { cap: 24 } } } }],
      metrics: [{ type: "divergence", column: "term_cap" }],
      plots: [{ type: "changes", metric: "divergence" }],
    },
    hasResult: true,
  },
  {
    slug: "sector-override-comparison",
    name: "sector-override-comparison",
    definition: {
      version: "2",
      name: "sector-override-comparison",
      description: "Forcing every record into the public-sector arm, next to the baseline.",
      flow: { entry: "loan:pipeline" },
      scenarios: [{ name: "all_public", at: "term/by_sector", overrides: { sector_code: 1 } }],
      metrics: [{ type: "branch_flows", nodes: "term/by_sector", outcome: "term_cap" }],
      plots: [{ type: "sankey", metric: "branch_flows", scenario: "all_public" }],
    },
  },
];
