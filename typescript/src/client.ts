/**
 * HTTP client for the agent-energy-budget service API.
 *
 * Uses the Fetch API (available natively in Node 18+, browsers, and Deno).
 * No external dependencies required.
 *
 * @example
 * ```ts
 * import { createAgentEnergyBudgetClient } from "@aumos/agent-energy-budget";
 *
 * const client = createAgentEnergyBudgetClient({ baseUrl: "http://localhost:8095" });
 *
 * // Check current budget
 * const budget = await client.getBudget("my-agent");
 * if (budget.ok) {
 *   console.log("Daily limit:", budget.data.daily_limit, "USD");
 * }
 *
 * // Record a completed LLM call
 * await client.trackCost({
 *   agent_id: "my-agent",
 *   model: "claude-haiku-4",
 *   input_tokens: 2048,
 *   output_tokens: 512,
 * });
 *
 * // Predict cost before making an LLM call
 * const forecast = await client.predictCost({
 *   agent_id: "my-agent",
 *   model: "gpt-4o-mini",
 *   prompt_text: "Summarize the report.",
 *   max_output_tokens: 256,
 * });
 * ```
 */

import type {
  ApiError,
  ApiResult,
  Budget,
  CostEntry,
  ModelPricing,
  PredictCostRequest,
  SetBudgetLimitRequest,
  TokenUsage,
  UsageReport,
  WorkloadForecast,
} from "./types.js";

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

/** Configuration options for the AgentEnergyBudgetClient. */
export interface AgentEnergyBudgetClientConfig {
  /** Base URL of the agent-energy-budget server (e.g. "http://localhost:8095"). */
  readonly baseUrl: string;
  /** Optional request timeout in milliseconds (default: 30000). */
  readonly timeoutMs?: number;
  /** Optional extra HTTP headers sent with every request. */
  readonly headers?: Readonly<Record<string, string>>;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function fetchJson<T>(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    clearTimeout(timeoutId);

    const body = await response.json() as unknown;

    if (!response.ok) {
      const errorBody = body as Partial<ApiError>;
      return {
        ok: false,
        error: {
          error: errorBody.error ?? "Unknown error",
          detail: errorBody.detail ?? "",
        },
        status: response.status,
      };
    }

    return { ok: true, data: body as T };
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    const message = err instanceof Error ? err.message : String(err);
    return {
      ok: false,
      error: { error: "Network error", detail: message },
      status: 0,
    };
  }
}

function buildHeaders(
  extraHeaders: Readonly<Record<string, string>> | undefined,
): Record<string, string> {
  return {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...extraHeaders,
  };
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-energy-budget service. */
export interface AgentEnergyBudgetClient {
  /**
   * Retrieve the current budget configuration for an agent.
   *
   * Returns daily, weekly, and monthly limits alongside the configured
   * degradation strategy, alert thresholds, and model preferences.
   *
   * @param agentId - The unique agent identifier.
   * @returns The Budget configuration record for this agent.
   */
  getBudget(agentId: string): Promise<ApiResult<Budget>>;

  /**
   * Record the actual cost of a completed LLM call.
   *
   * Persists the call record to JSONL storage and fires any alert
   * thresholds that have been crossed as a result of this spend.
   *
   * @param entry - Cost record with agent, model, token counts, and optional cost.
   * @returns The persisted TokenUsage record with calculated cost_usd.
   */
  trackCost(entry: CostEntry): Promise<ApiResult<TokenUsage>>;

  /**
   * Predict the cost of an LLM call before it is executed.
   *
   * Uses the pricing tables and token estimation heuristics to return
   * an upper-bound cost estimate with a confidence score.
   *
   * @param request - Prediction request with model, prompt text, and token cap.
   * @returns WorkloadForecast with estimated tokens, cost, and confidence.
   */
  predictCost(request: PredictCostRequest): Promise<ApiResult<WorkloadForecast>>;

  /**
   * Update the budget limits for an agent.
   *
   * Supports updating daily, weekly, and monthly limits independently.
   * Only fields provided in the request are modified; omitted fields retain
   * their current values.
   *
   * @param request - Budget limit update payload.
   * @returns The updated Budget configuration record.
   */
  setBudgetLimit(request: SetBudgetLimitRequest): Promise<ApiResult<Budget>>;

  /**
   * Retrieve a consolidated usage report for an agent across all periods.
   *
   * Returns daily, weekly, and monthly BudgetStatus snapshots plus a
   * lifetime total spend.
   *
   * @param agentId - The unique agent identifier.
   * @param options - Optional filter by period ("daily" | "weekly" | "monthly").
   * @returns UsageReport with per-period snapshots and lifetime totals.
   */
  getUsageReport(
    agentId: string,
    options?: { period?: "daily" | "weekly" | "monthly" },
  ): Promise<ApiResult<UsageReport>>;

  /**
   * Retrieve pricing information for a specific model.
   *
   * Supports fuzzy model name resolution including aliases (e.g. "haiku"
   * resolves to "claude-haiku-4"). Returns null in the data field if the
   * model cannot be resolved.
   *
   * @param model - Model identifier string (exact or alias).
   * @returns ModelPricing with per-million token rates, tier, and context window.
   */
  getModelPricing(model: string): Promise<ApiResult<ModelPricing>>;
}

// ---------------------------------------------------------------------------
// Client factory
// ---------------------------------------------------------------------------

/**
 * Create a typed HTTP client for the agent-energy-budget service.
 *
 * @param config - Client configuration including base URL.
 * @returns An AgentEnergyBudgetClient instance.
 */
export function createAgentEnergyBudgetClient(
  config: AgentEnergyBudgetClientConfig,
): AgentEnergyBudgetClient {
  const { baseUrl, timeoutMs = 30_000, headers: extraHeaders } = config;
  const baseHeaders = buildHeaders(extraHeaders);

  return {
    async getBudget(agentId: string): Promise<ApiResult<Budget>> {
      return fetchJson<Budget>(
        `${baseUrl}/budgets/${encodeURIComponent(agentId)}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async trackCost(entry: CostEntry): Promise<ApiResult<TokenUsage>> {
      return fetchJson<TokenUsage>(
        `${baseUrl}/costs`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(entry),
        },
        timeoutMs,
      );
    },

    async predictCost(
      request: PredictCostRequest,
    ): Promise<ApiResult<WorkloadForecast>> {
      return fetchJson<WorkloadForecast>(
        `${baseUrl}/costs/predict`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(request),
        },
        timeoutMs,
      );
    },

    async setBudgetLimit(
      request: SetBudgetLimitRequest,
    ): Promise<ApiResult<Budget>> {
      return fetchJson<Budget>(
        `${baseUrl}/budgets/${encodeURIComponent(request.agent_id)}/limits`,
        {
          method: "PATCH",
          headers: baseHeaders,
          body: JSON.stringify(request),
        },
        timeoutMs,
      );
    },

    async getUsageReport(
      agentId: string,
      options?: { period?: "daily" | "weekly" | "monthly" },
    ): Promise<ApiResult<UsageReport>> {
      const params = new URLSearchParams();
      if (options?.period !== undefined) {
        params.set("period", options.period);
      }
      const queryString = params.toString();
      const url = queryString
        ? `${baseUrl}/reports/${encodeURIComponent(agentId)}?${queryString}`
        : `${baseUrl}/reports/${encodeURIComponent(agentId)}`;
      return fetchJson<UsageReport>(
        url,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async getModelPricing(model: string): Promise<ApiResult<ModelPricing>> {
      return fetchJson<ModelPricing>(
        `${baseUrl}/pricing/${encodeURIComponent(model)}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },
  };
}

