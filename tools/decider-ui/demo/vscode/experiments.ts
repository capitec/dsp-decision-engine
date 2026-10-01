// Mock stand-ins for the project-owned `experiments/<slug>/experiment.yaml` assets — just enough
// to drive the sidebar's Experiments & Results pane (name + search) before anything opens one.

export interface MockExperiment {
  slug: string;
  name: string;
}

export const EXPERIMENTS: MockExperiment[] = [
  { slug: "affordability-verdict-drift", name: "affordability-verdict-drift" },
  { slug: "term-cap-sensitivity", name: "term-cap-sensitivity" },
  { slug: "sector-override-comparison", name: "sector-override-comparison" },
];
