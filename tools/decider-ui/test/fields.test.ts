import { afterEach, describe, expect, it } from "vitest";
import { fieldOf, formatValue, setFields } from "../src/model/protocol";

describe("formatting by what a value measures", () => {
  afterEach(() => setFields(undefined));

  it("never groups thousands", () => {
    expect(formatValue(4000)).toBe("4000");
    expect(formatValue(58113.066)).toBe("58113.07");
  });

  it("guesses from the name when nothing is declared, and says so", () => {
    expect(formatValue(9000, "net_income")).toBe("R 9000.00");
    expect(formatValue(0.252, "pl_rate")).toBe("25.2%");
    expect(formatValue(36, "term_months")).toBe("36 months");
    expect(fieldOf("net_income")?.assumed).toBe(true);
    expect(fieldOf("score")).toBeUndefined();
  });

  it("uses declared metadata over the name", () => {
    setFields({ net_income: { kind: "value" }, balance: { kind: "money", symbol: "R", cents: true }, term: { kind: "duration", unit: "months" } });
    expect(formatValue(9000, "net_income")).toBe("9000");
    expect(formatValue(123456, "balance")).toBe("R 1234.56");
    expect(formatValue(1, "term")).toBe("1 month");
    expect(fieldOf("balance")?.assumed).toBeUndefined();
  });
});
