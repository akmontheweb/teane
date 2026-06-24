You are a Principal Software Architect. Generate a complete, build-ready Architecture Decision Record + Architecture Design Document for the project below. Be exhaustive — downstream code generation and the production-readiness review will depend on this document. Prefer structured depth (every section has a required schema) over prose length.

## Architecture Document

### 1. System Context (C4 Level 1)
- One-sentence purpose; primary users and their goals
- External systems this system depends on (with direction: read / write / both)
- External systems that depend on this system

### 2. Container Diagram (C4 Level 2)
For each deployable unit (web app, API gateway, worker, database, cache, message broker, CDN, edge function, batch job):
- Container name + responsibility (one line)
- Protocols it speaks (inbound + outbound)
- Tech stack (language, framework, runtime version)
- Statefulness (stateless / sticky session / persistent)
- Public exposure (internet-facing / private / VPC-only)

### 3. Component Diagram (C4 Level 3)
For each container, list its internal components/modules:
- Component name + responsibility
- Public API exposed to peers
- Owned data / external state read
- Dependency graph (which components call which)

### 4. Data Model & Ownership
- ER diagram (textual) — entities, attributes, relationships, cardinalities
- Per table: PK, FKs, unique constraints, indexes (B-tree / hash / GIN / composite), partitioning strategy
- **Single-writer owner** per dataset (which service is authoritative?)
- Replication direction (master-slave / multi-master / leaderless), conflict-resolution policy
- Connection pooling per service; read-replica routing rules

### 5. Component Interfaces & Contracts
Per service-to-service edge:
- Synchronous (REST / gRPC / GraphQL) vs asynchronous (queue / stream / pub-sub)
- Request envelope, serialization format, schema location
- Default timeout, retry policy, idempotency-key window
- Sequence diagram (textual) for any flow with > 2 hops
- Contract evolution: versioning scheme, deprecation policy, breaking-change rules

### 6. Trust Boundaries & Security Zones
- Network zones (public edge / DMZ / internal / restricted) — what traverses each boundary
- AuthN method at each boundary (mTLS, JWT, signed-request, SPIFFE)
- AuthZ enforcement point (gateway / service / data-tier)
- Encryption: in-transit (TLS version, cipher suite policy) and at-rest (KMS, envelope encryption)
- Tokenization / redaction rules at boundary crossings
- Threat model (STRIDE) — per asset, the mitigation deployed for each category

### 7. External Dependencies & Rate Limits
Per third-party service:
- Purpose, criticality (P0/P1/P2)
- Provider SLA, our consumption rate, vendor rate limits (RPS, RPM, daily quota)
- Throttling algorithm (token bucket, sliding window), retry-after handling
- Fallback strategy when degraded or down (cache, queue, degraded UX, fail-closed)
- Vendor-lock-in posture (abstraction layer? direct coupling?)

### 8. Storage Topology
- Per service: durable storage choice, why (rationale: scale, query pattern, consistency model)
- Volume strategy (named volumes vs bind mounts vs tmpfs)
- Cross-instance shared state (NFS / EFS / object store) — when needed and how avoided otherwise
- Caching tier (Redis / Memcached / app-local) — invalidation strategy, TTLs, stampede protection
- Backup target, frequency, encryption, off-region copy policy, restore-test cadence

### 9. Secrets & Configuration Management
- Secrets store (Vault / AWS Secrets Manager / Doppler / SOPS-encrypted git)
- Injection method (env var, file mount, sidecar)
- Per-secret rotation cadence, revocation procedure
- Runtime config source-of-truth (env / ConfigMap / feature-flag service), hot-reload semantics, audit trail

### 10. Deployment Topology
- Regions, availability zones, edge/core tiers (CDN, edge compute, regional, central)
- Active-active vs active-passive; failover trigger; expected failover time
- Tenancy model (single-tenant / pooled multi-tenant / silo'd multi-tenant) and isolation boundaries
- Per-environment topology (dev / staging / prod / DR) and parity rules

### 11. Scaling & Performance Budgets
- Per service: horizontal vs vertical scaling axes; autoscaling triggers (CPU / memory / queue depth / custom SLI) with min/max replicas
- Per-endpoint latency budget (P50 / P95 / P99), throughput target, burst headroom
- Cold-start mitigation (warm pool size, provisioned concurrency, lazy init)
- Connection-pool sizing, thread-pool sizing

### 12. Failure Domains & Resilience Patterns
- Failure domains: AZ loss, region loss, dependency loss, pool exhaustion — for each: what continues, what degrades, what fails
- Resilience patterns deployed: circuit breaker, bulkhead, timeout cascade, retry budget, hedging, backpressure, dead-letter queues
- Chaos / game-day cadence; pre-launch failure-injection requirements

### 13. Observability & Alerting
- Structured-log schema (required fields, format), log aggregation pipeline, retention
- Metrics backend; coverage of golden signals per service (latency, traffic, errors, saturation)
- Distributed tracing (OpenTelemetry), trace-id propagation rules, sampling policy
- Alert taxonomy (page / ticket / silent), on-call rotation, runbook linkage per alert

### 14. CI/CD & Release Strategy
- Build triggers (push / PR / tag)
- Required gates: lint, unit, integration, security scan, SBOM, license check, contract test
- Release strategy per service: rolling / blue-green / canary / ring-based; automated rollback criteria
- Environment promotion path; required approvals; change-management/audit linkage

### 15. Data Lifecycle (Migration, Backup, Archival)
- Schema migration tooling and forward/backward compatibility rules
- Online migration patterns (expand–contract, dual-write, shadow read)
- Backup cadence, RPO / RTO targets, restore-test plan
- Archival tier and trigger; legal-hold mechanism; bulk export / portability obligations

### 16. Technology Stack
- Languages, frameworks, runtimes (with pinned versions)
- Databases, caches, message brokers (with pinned versions)
- Infrastructure / orchestration (Kubernetes, ECS, Nomad, bare-VM)
- Build & dependency tooling

### 17. Key Architecture Decisions (ADRs)
For each non-obvious choice:
- **ADR-NNN**: Title
  - Context: what problem are we solving?
  - Decision: what we chose
  - Alternatives considered (with one-line tradeoff each)
  - Consequences (positive and negative)
  - Status (Proposed / Accepted / Superseded)

### 18. Explicit Non-Goals
- What the system intentionally does NOT do, and why
- Boundaries we are not pushing in this iteration
- Anti-features (things we choose not to build)

---

Output as a single well-formatted Markdown document. Use the project file structure to ground every claim — when a section's information cannot be inferred from the code, state the conservative industry default explicitly and call it out as an open decision (e.g. "ADR-OPEN-NN: choose X vs Y"). Avoid generic placeholders such as "appropriate framework" or "industry-standard auth" — be specific.
