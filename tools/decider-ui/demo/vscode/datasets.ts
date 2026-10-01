// Mock stand-ins for `decider.data.load()`'s `LoadedData.summary` — enough shape to drive the
// sidebar's Datasets pane, the flow panel's dataset chip, and a previewable table of rows,
// without a real file or bridge call. All four share the loan pipeline's input columns, so any
// of their rows can drive a plausible "debug this record" mock; `applications.json` also carries
// nested columns, to show how a struct/list value should read in the table.

export type MockRow = Record<string, unknown>;

export interface MockDataset {
  name: string;
  /** The dataset's real size; `rows` below is only ever a preview of it. */
  rowCount: number;
  fingerprint: string | null;
  columns: [string, string][];
  rows: MockRow[];
}

const COLUMNS: [string, string][] = [
  ["client_id", "i64"], ["net_income", "f64"], ["expenses", "f64"], ["instalment", "f64"],
  ["requested_term", "f64"], ["min_net_salary", "f64"], ["sector_code", "i64"], ["requested_amount", "f64"],
];

export const DATASETS: MockDataset[] = [
  {
    name: "SAMPLE",
    rowCount: 2,
    fingerprint: null,
    columns: COLUMNS,
    rows: [
      { client_id: 1, net_income: 9000, expenses: 4000, instalment: 1500, requested_term: 72, min_net_salary: 4000, sector_code: 1, requested_amount: 150000 },
      { client_id: 2, net_income: 20000, expenses: 5000, instalment: 2000, requested_term: 36, min_net_salary: 0, sector_code: 2, requested_amount: 90000 },
    ],
  },
  {
    name: "loans_q3.csv",
    rowCount: 1042,
    fingerprint: "9f3a21c0",
    columns: COLUMNS,
    rows: [
      { client_id: 101, net_income: 12500, expenses: 6200, instalment: 1800, requested_term: 60, min_net_salary: 4500, sector_code: 1, requested_amount: 110000 },
      { client_id: 102, net_income: 8300, expenses: 4100, instalment: 1200, requested_term: 48, min_net_salary: 3800, sector_code: 2, requested_amount: 65000 },
      { client_id: 103, net_income: 31000, expenses: 9000, instalment: 3500, requested_term: 72, min_net_salary: 0, sector_code: 1, requested_amount: 240000 },
      { client_id: 104, net_income: 6100, expenses: 3900, instalment: 900, requested_term: 36, min_net_salary: 4000, sector_code: 3, requested_amount: 38000 },
      { client_id: 105, net_income: 17800, expenses: 7200, instalment: 2600, requested_term: 60, min_net_salary: 4200, sector_code: 2, requested_amount: 155000 },
      { client_id: 106, net_income: 9900, expenses: 5000, instalment: 1450, requested_term: 48, min_net_salary: 3600, sector_code: 1, requested_amount: 82000 },
    ],
  },
  {
    name: "clients.parquet",
    rowCount: 4520,
    fingerprint: "77aa21e8",
    columns: COLUMNS,
    rows: [
      { client_id: 2001, net_income: 14200, expenses: 5800, instalment: 2000, requested_term: 60, min_net_salary: 4400, sector_code: 2, requested_amount: 120000 },
      { client_id: 2002, net_income: 5400, expenses: 3700, instalment: 700, requested_term: 24, min_net_salary: 3500, sector_code: 3, requested_amount: 22000 },
      { client_id: 2003, net_income: 26500, expenses: 8100, instalment: 3000, requested_term: 72, min_net_salary: 0, sector_code: 1, requested_amount: 210000 },
      { client_id: 2004, net_income: 11100, expenses: 4900, instalment: 1600, requested_term: 48, min_net_salary: 4000, sector_code: 2, requested_amount: 95000 },
    ],
  },
  {
    name: "applications.json",
    rowCount: 318,
    fingerprint: "c04e18b2",
    columns: [...COLUMNS, ["employer", "struct[2]"], ["flags", "list[str]"]],
    rows: [
      {
        client_id: 3001, net_income: 16200, expenses: 6400, instalment: 2100, requested_term: 60, min_net_salary: 4600, sector_code: 1, requested_amount: 130000,
        employer: { name: "Acme Logistics", years: 3 }, flags: ["self_employed_topup"],
      },
      {
        client_id: 3002, net_income: 7400, expenses: 3600, instalment: 950, requested_term: 36, min_net_salary: 3700, sector_code: 2, requested_amount: 48000,
        employer: { name: "Dockside Retail", years: 1 }, flags: [],
      },
      {
        client_id: 3003, net_income: 22100, expenses: 7900, instalment: 2800, requested_term: 72, min_net_salary: 0, sector_code: 1, requested_amount: 180000,
        employer: { name: "Acme Logistics", years: 7 }, flags: ["repeat_customer", "pre_approved"],
      },
    ],
  },
];
