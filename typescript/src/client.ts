/**
 * HTTP client for the agent-energy-budget service API.
 *
 * Delegates all HTTP transport to `@aumos/sdk-core` which provides
 * automatic retry with exponential back-off, timeout management via
 * `AbortSignal.timeout`, interceptor support, and a typed error hierarchy.
 *
 * The public-facing `ApiResult<T>` envelope is preserved for full
 * backward compatibility with existing callers.
 *
 * @example
 * ```ts
 * import { createAgentEnergyBudgetClient } from "@aumos/agent-energy-budget";
 *
 * const client = createAgentEnergyBudgetClient({ baseUrl: "http://localhost:8095" });
 *
 * const budget = await client.getBudget("my-agent");
 * if (budget.ok) {
 *   console.log("Daily limit:", budget.data.daily_limit, "USD");
 * }
 *
 * await client.trackCost({
 *   agent_id: "my-agent",
 *   model: "claude-haiku-4",
 *   input_tokens: 2048,
 *   output_tokens: 512,
 * });
 * ```
 */

import {
  createHttpClient,
  HttpError,
  NetworkError,
  TimeoutError,
  AumosError,
  type HttpClient,
} from "@aumos/sdk-core";

import type {
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
// Internal adapter
// ---------------------------------------------------------------------------

async function callApi<T>(
  operation: () => Promise<{ readonly data: T; readonly status: number }>,
): Promise<ApiResult<T>> {
  try {
    const response = await operation();
    return { ok: true, data: response.data };
  } catch (error: unknown) {
    if (error instanceof HttpError) {
      return {
        ok: false,
        error: { error: error.message, detail: String(error.body ?? "") },
        status: error.statusCode,
      };
    }
    if (error instanceof TimeoutError) {
      return {
        ok: false,
        error: { error: "Request timed out", detail: error.message },
        status: 0,
      };
    }
    if (error instanceof NetworkError) {
      return {
        ok: false,
        error: { error: "Network error", detail: error.message },
        status: 0,
      };
    }
    if (error instanceof AumosError) {
      return {
        ok: false,
        error: { error: error.code, detail: error.message },
        status: error.statusCode ?? 0,
      };
    }
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      error: { error: "Unexpected error", detail: message },
      status: 0,
    };
  }
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-energy-budget service. */
export interface AgentEnergyBudgetClient {
  /**
   * Retrieve the current budget configuration for an agent.
   *
   * @param agentId - The unique agent identifier.
   * @returns The Budget configuration record for this agent.
   */
  getBudget(agentId: string): Promise<ApiResult<Budget>>;

  /**
   * Record the actual cost of a completed LLM call.
   *
   * @param entry - Cost record with agent, model, token counts, and optional cost.
   * @returns The persisted TokenUsage record with calculated cost_usd.
   */
  trackCost(entry: CostEntry): Promise<ApiResult<TokenUsage>>;

  /**
   * Predict the cost of an LLM call before it is executed.
   *
   * @param request - Prediction request with model, prompt text, and token cap.
   * @returns WorkloadForecast with estimated tokens, cost, and confidence.
   */
  predictCost(request: PredictCostRequest): Promise<ApiResult<WorkloadForecast>>;

  /**
   * Update the budget limits for an agent.
   *
   * @param request - Budget limit update payload.
   * @returns The updated Budget configuration record.
   */
  setBudgetLimit(request: SetBudgetLimitRequest): Promise<ApiResult<Budget>>;

  /**
   * Retrieve a consolidated usage report for an agent across all periods.
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
  const http: HttpClient = createHttpClient({
    baseUrl: config.baseUrl,
    timeout: config.timeoutMs ?? 30_000,
    defaultHeaders: config.headers,
  });

  return {
    getBudget(agentId: string): Promise<ApiResult<Budget>> {
      return callApi(() =>
        http.get<Budget>(`/budgets/${encodeURIComponent(agentId)}`),
      );
    },

    trackCost(entry: CostEntry): Promise<ApiResult<TokenUsage>> {
      return callApi(() => http.post<TokenUsage>("/costs", entry));
    },

    predictCost(request: PredictCostRequest): Promise<ApiResult<WorkloadForecast>> {
      return callApi(() => http.post<WorkloadForecast>("/costs/predict", request));
    },

    setBudgetLimit(request: SetBudgetLimitRequest): Promise<ApiResult<Budget>> {
      return callApi(() =>
        http.patch<Budget>(
          `/budgets/${encodeURIComponent(request.agent_id)}/limits`,
          request,
        ),
      );
    },

    getUsageReport(
      agentId: string,
      options?: { period?: "daily" | "weekly" | "monthly" },
    ): Promise<ApiResult<UsageReport>> {
      const queryParams: Record<string, string> = {};
      if (options?.period !== undefined) queryParams["period"] = options.period;
      return callApi(() =>
        http.get<UsageReport>(
          `/reports/${encodeURIComponent(agentId)}`,
          { queryParams },
        ),
      );
    },

    getModelPricing(model: string): Promise<ApiResult<ModelPricing>> {
      return callApi(() =>
        http.get<ModelPricing>(`/pricing/${encodeURIComponent(model)}`),
      );
    },
  };
}
