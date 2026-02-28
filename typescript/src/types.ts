/**
 * TypeScript interfaces for the agent-energy-budget service.
 *
 * Mirrors the Pydantic models and dataclasses defined in:
 *   agent_energy_budget.budget.config
 *   agent_energy_budget.budget.tracker
 *   agent_energy_budget.budget.alerts
 *   agent_energy_budget.pricing.tables
 *   agent_energy_budget.estimator.cost_estimator
 *
 * All interfaces use readonly fields to match Python's frozen dataclasses.
 */

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/**
 * Action to take when a budget limit is approached or exceeded.
 * Maps to DegradationStrategy enum in agent_energy_budget.budget.config.
 */
export type DegradationStrategy =
  | "model_downgrade"
  | "token_reduction"
  | "block_with_error"
  | "cached_fallback";

/**
 * LLM provider identifier.
 * Maps to ProviderName enum in agent_energy_budget.pricing.tables.
 */
export type ProviderName =
  | "anthropic"
  | "openai"
  | "google"
  | "mistral"
  | "meta"
  | "deepseek"
  | "custom";

/**
 * Quality/cost tier classification for models.
 * Maps to ModelTier enum in agent_energy_budget.pricing.tables.
 *
 * nano      — ultra-cheap, small context, fast
 * efficient — good value, solid quality
 * standard  — mainstream flagship-class models
 * premium   — top capability, highest cost
 */
export type ModelTier = "nano" | "efficient" | "standard" | "premium";

/**
 * Alert severity level.
 * Maps to AlertLevel enum in agent_energy_budget.budget.alerts.
 */
export type AlertLevel = "warning" | "critical" | "exhausted";

// ---------------------------------------------------------------------------
// TokenUsage
// ---------------------------------------------------------------------------

/**
 * Token consumption record for a single LLM call.
 * Mirrors the _CallRecord internal dataclass in agent_energy_budget.budget.tracker.
 */
export interface TokenUsage {
  /** Agent that made this call. */
  readonly agent_id: string;
  /** Model identifier used for this call. */
  readonly model: string;
  /** Number of input (prompt) tokens consumed. */
  readonly input_tokens: number;
  /** Number of output (completion) tokens generated. */
  readonly output_tokens: number;
  /** Actual cost in USD for this call. */
  readonly cost_usd: number;
  /** ISO-8601 UTC timestamp when this call was recorded. */
  readonly recorded_at: string;
}

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

/**
 * Alert threshold configuration as utilisation percentages.
 * Maps to AlertThresholds in agent_energy_budget.budget.config.
 */
export interface AlertThresholds {
  /** First alert threshold (default 50%). */
  readonly warning: number;
  /** Second alert threshold (default 80%). */
  readonly critical: number;
  /** Third alert threshold (default 100%). */
  readonly exhausted: number;
}

/**
 * Model selection preferences for degradation and estimation.
 * Maps to ModelPreferences in agent_energy_budget.budget.config.
 */
export interface ModelPreferences {
  /** Ordered list of model IDs to try first (most preferred first). */
  readonly preferred_models: readonly string[];
  /** Model to use when all others are exhausted or over budget. */
  readonly fallback_model: string;
  /** Model IDs that must never be used. */
  readonly blocked_models: readonly string[];
  /** When true, restrict model selection to vision-capable models only. */
  readonly require_vision: boolean;
}

/**
 * Top-level budget configuration for a single agent or agent group.
 * Maps to BudgetConfig in agent_energy_budget.budget.config.
 */
export interface Budget {
  /** Unique identifier for the agent this budget belongs to. */
  readonly agent_id: string;
  /** Maximum USD spend per calendar day (0 = disabled). */
  readonly daily_limit: number;
  /** Maximum USD spend per calendar week Mon-Sun (0 = disabled). */
  readonly weekly_limit: number;
  /** Maximum USD spend per calendar month (0 = disabled). */
  readonly monthly_limit: number;
  /** Strategy to apply when a limit is breached. */
  readonly degradation_strategy: DegradationStrategy;
  /** Utilisation percentage levels that trigger alerts. */
  readonly alert_thresholds: AlertThresholds;
  /** Model selection configuration for degradation. */
  readonly model_preferences: ModelPreferences;
  /** ISO currency code for display purposes (default "USD"). */
  readonly currency: string;
  /** Arbitrary key-value metadata attached to this budget. */
  readonly tags: Readonly<Record<string, string>>;
}

// ---------------------------------------------------------------------------
// BudgetStatus (usage snapshot)
// ---------------------------------------------------------------------------

/**
 * Snapshot of current budget usage for a single period.
 * Maps to BudgetStatus dataclass in agent_energy_budget.budget.tracker.
 */
export interface BudgetStatus {
  /** The agent whose budget this describes. */
  readonly agent_id: string;
  /** Period label: "daily", "weekly", or "monthly". */
  readonly period: "daily" | "weekly" | "monthly";
  /** Budget cap in USD for this period. */
  readonly limit_usd: number;
  /** Amount spent so far this period in USD. */
  readonly spent_usd: number;
  /** Amount remaining (limit - spent); may be negative if overspent. */
  readonly remaining_usd: number;
  /** Percentage of budget consumed (0-100+). */
  readonly utilisation_pct: number;
  /** Number of LLM calls recorded this period. */
  readonly call_count: number;
  /** Average cost per call in USD. */
  readonly avg_cost_per_call: number;
}

// ---------------------------------------------------------------------------
// CostEntry (track cost request)
// ---------------------------------------------------------------------------

/**
 * Request payload for recording actual LLM call costs.
 */
export interface CostEntry {
  /** Agent that made the call. */
  readonly agent_id: string;
  /** Model identifier used. */
  readonly model: string;
  /** Actual input tokens consumed. */
  readonly input_tokens: number;
  /** Actual output tokens generated. */
  readonly output_tokens: number;
  /** Actual cost in USD (if null, calculated from pricing tables). */
  readonly cost_usd?: number;
}

// ---------------------------------------------------------------------------
// ModelPricing
// ---------------------------------------------------------------------------

/**
 * Per-model pricing in USD per million tokens.
 * Maps to ModelPricing dataclass in agent_energy_budget.pricing.tables.
 */
export interface ModelPricing {
  /** Canonical model identifier string. */
  readonly model: string;
  /** The provider that operates this model. */
  readonly provider: ProviderName;
  /** Quality/cost tier classification. */
  readonly tier: ModelTier;
  /** Cost in USD for one million input (prompt) tokens. */
  readonly input_per_million: number;
  /** Cost in USD for one million output (completion) tokens. */
  readonly output_per_million: number;
  /** Maximum context window in tokens (0 = unknown). */
  readonly context_window: number;
  /** Whether the model accepts image inputs. */
  readonly supports_vision: boolean;
}

// ---------------------------------------------------------------------------
// BudgetAlert
// ---------------------------------------------------------------------------

/**
 * Immutable record of a fired budget alert event.
 * Maps to AlertEvent dataclass in agent_energy_budget.budget.alerts.
 */
export interface BudgetAlert {
  /** The agent whose budget triggered the alert. */
  readonly agent_id: string;
  /** Alert severity level. */
  readonly level: AlertLevel;
  /** Current utilisation percentage at the time of the alert. */
  readonly utilisation_pct: number;
  /** Amount spent in USD at the time of the alert. */
  readonly spent_usd: number;
  /** Budget limit in USD that was approached or exceeded. */
  readonly limit_usd: number;
  /** Budget period label ("daily", "weekly", or "monthly"). */
  readonly period: string;
  /** Human-readable description of the alert condition. */
  readonly message: string;
  /** ISO-8601 UTC timestamp when the alert was fired. */
  readonly fired_at: string;
}

// ---------------------------------------------------------------------------
// WorkloadForecast (cost prediction)
// ---------------------------------------------------------------------------

/**
 * Pre-execution cost estimate for an LLM call.
 * Maps to CostEstimate dataclass in agent_energy_budget.estimator.cost_estimator.
 */
export interface WorkloadForecast {
  /** Model identifier used for the estimate. */
  readonly model: string;
  /** Estimated number of input (prompt) tokens. */
  readonly estimated_input_tokens: number;
  /** Estimated number of output (completion) tokens. */
  readonly estimated_output_tokens: number;
  /** Estimated total cost in USD. */
  readonly estimated_cost_usd: number;
  /** Confidence score for the estimate in [0.0, 1.0]. */
  readonly confidence: number;
}

// ---------------------------------------------------------------------------
// PredictCostRequest
// ---------------------------------------------------------------------------

/**
 * Request payload for predicting the cost of an LLM call before execution.
 */
export interface PredictCostRequest {
  /** Agent identifier requesting the cost prediction. */
  readonly agent_id: string;
  /** Model to use for cost estimation. */
  readonly model: string;
  /** Prompt text to estimate input tokens from. */
  readonly prompt_text: string;
  /** Maximum output tokens (used as worst-case upper bound). */
  readonly max_output_tokens?: number;
}

// ---------------------------------------------------------------------------
// SetBudgetLimitRequest
// ---------------------------------------------------------------------------

/**
 * Request payload for updating budget limits for an agent.
 */
export interface SetBudgetLimitRequest {
  /** Agent identifier to update limits for. */
  readonly agent_id: string;
  /** New daily limit in USD (0 = disabled, omit to leave unchanged). */
  readonly daily_limit?: number;
  /** New weekly limit in USD (0 = disabled, omit to leave unchanged). */
  readonly weekly_limit?: number;
  /** New monthly limit in USD (0 = disabled, omit to leave unchanged). */
  readonly monthly_limit?: number;
  /** Degradation strategy to apply on limit breach. */
  readonly degradation_strategy?: DegradationStrategy;
}

// ---------------------------------------------------------------------------
// ThrottleConfig
// ---------------------------------------------------------------------------

/**
 * Throttling configuration applied when budget limits approach exhaustion.
 * Maps to options in agent_energy_budget.budget.config.
 */
export interface ThrottleConfig {
  /** Agent identifier this throttle configuration applies to. */
  readonly agent_id: string;
  /** Whether throttling is currently active for this agent. */
  readonly throttle_active: boolean;
  /** Maximum calls per minute allowed under throttle. */
  readonly max_calls_per_minute: number;
  /** Maximum input tokens per call under throttle. */
  readonly max_input_tokens_per_call: number;
  /** Maximum output tokens per call under throttle. */
  readonly max_output_tokens_per_call: number;
  /** ISO-8601 UTC timestamp when throttling was last applied. */
  readonly throttle_applied_at: string | null;
}

// ---------------------------------------------------------------------------
// UsageReport
// ---------------------------------------------------------------------------

/**
 * Aggregated usage report across all periods for an agent.
 */
export interface UsageReport {
  /** Agent identifier this report covers. */
  readonly agent_id: string;
  /** Daily usage snapshot. */
  readonly daily: BudgetStatus;
  /** Weekly usage snapshot. */
  readonly weekly: BudgetStatus;
  /** Monthly usage snapshot. */
  readonly monthly: BudgetStatus;
  /** Total USD spent across all recorded history. */
  readonly lifetime_spend_usd: number;
  /** ISO-8601 UTC timestamp when this report was generated. */
  readonly generated_at: string;
}

// ---------------------------------------------------------------------------
// API result wrapper (shared pattern)
// ---------------------------------------------------------------------------

/** Standard error payload returned by the agent-energy-budget API. */
export interface ApiError {
  readonly error: string;
  readonly detail: string;
}

/** Result type for all client operations. */
export type ApiResult<T> =
  | { readonly ok: true; readonly data: T }
  | { readonly ok: false; readonly error: ApiError; readonly status: number };
