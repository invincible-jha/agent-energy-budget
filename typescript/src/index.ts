/**
 * @aumos/agent-energy-budget
 *
 * TypeScript client for the AumOS agent-energy-budget service.
 * Provides HTTP client and type definitions for token cost tracking,
 * budget enforcement, cost prediction, and model pricing lookup.
 */

// Client and configuration
export type { AgentEnergyBudgetClient, AgentEnergyBudgetClientConfig } from "./client.js";
export { createAgentEnergyBudgetClient } from "./client.js";

// Core types
export type {
  DegradationStrategy,
  ProviderName,
  ModelTier,
  AlertLevel,
  TokenUsage,
  AlertThresholds,
  ModelPreferences,
  Budget,
  BudgetStatus,
  CostEntry,
  ModelPricing,
  BudgetAlert,
  WorkloadForecast,
  PredictCostRequest,
  SetBudgetLimitRequest,
  ThrottleConfig,
  UsageReport,
  ApiError,
  ApiResult,
} from "./types.js";
