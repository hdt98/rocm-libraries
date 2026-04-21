# Federated Merge Queues for rocm-libraries

- Contributors: Samuel Reeder
- **Status**: First draft
- **Target Branch**: `develop` (only)

## 1. Executive Summary

This RFC proposes a custom, in-repo merge queue for rocm-libraries that serializes merges across coupled components on `develop`. Each opted-in component gets its own FIFO queue. A PR enters every queue of every component it depends on, and merges only when it is at the head of all of them. Authors trigger the queue with a `/merge` PR comment; an in-repo workflow squash-merges PRs as they reach the front.

Initial scope is the hipDNN ecosystem: `hipdnn` core, four providers (`miopen-provider`, `hipblaslt-provider`, `hip-kernel-provider`, `fusilli-provider`), and `integration-tests`. The queue is opt-in — other rocm-libraries subprojects are unaffected — starts optional (no branch-protection changes), and is designed so any project can join later by adding two centralized config entries.

## 2. Problem Statement

- **Provider/core coupling.** A change to hipDNN core can break any provider; a change to a provider can silently regress hipDNN integration. Today, nothing serializes these merges, so two PRs touching coupled code can both pass CI in isolation and break `develop` together.
- **CI capacity.** Parallel un-rebased merges thrash `develop` and force needless re-runs of expensive provider builds.
- **Developer overhead.** A reviewer should not have to manually coordinate merge order across components.

We want a system that serializes safely by default, is cheap to opt into, and stays out of contributors' way.

## 3. Goals and Non-Goals

### Goals
- One FIFO queue per component: `hipdnn`, four providers, `integration-tests`.
- Cross-blocking that follows dependencies (see [§4.2](#42-path--queue-mapping)). A PR is blocked by every PR ahead of it in any queue it belongs to.
- Opt-in per project. Subprojects outside the hipDNN ecosystem are unaffected unless they opt in.
- `develop` is the only managed target branch.
- Start optional; provide a clear path to "required" via branch protection.
- External contributors can use the queue with the same safeguards as anyone else.

### Non-Goals
- Replacing existing CI workflows.
- Other rocm-libraries projects in v1.
- Batched merges (`batch_size > 1`) — deferred.
- Release or feature branches — `develop` only.

## 4. Design

### 4.1 Queues

Six FIFO queues, one per component:

- `hipdnn`
- `miopen-provider`
- `hipblaslt-provider`
- `hip-kernel-provider`
- `fusilli-provider`
- `integration-tests`

Each PR enters zero or more queues based on the paths it touches. A PR merges only when at the head of **every** queue it belongs to.

### 4.2 Path → Queue mapping

The mapping encodes a single rule: **a PR enters its own queue plus the queue of every downstream component** — those that depend on it and could break if it changes. This ensures the PR serializes with any in-flight work in components it could affect.

| Path changed                          | Queues entered                                                                                          | Why                                                                                                                                                  |
|---------------------------------------|---------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `projects/hipdnn/**`                  | `hipdnn`, `miopen-provider`, `hipblaslt-provider`, `hip-kernel-provider`, `fusilli-provider`, `integration-tests` (all 6) | Core is upstream of every other component — all providers and integration-tests are downstream. A core PR enters all six queues so it serializes with everything. |
| `dnn-providers/<provider>/**`         | `<provider>`, `integration-tests`                                                                       | Each provider's only downstream component is `integration-tests`. Does **not** enter `hipdnn`; core PRs enter the provider's queue (because providers are downstream of core), which handles that serialization. Providers serialize with each other only indirectly via the shared `integration-tests` queue. |
| `dnn-providers/integration-tests/**`  | `integration-tests`                                                                                    | Integration-tests has no downstream components, so it enters only its own queue. Upstream components (providers, core) already enter `integration-tests`, so serialization happens through the shared queue. |
| Anything else                         | none                                                                                                    | Project not opted in.                                                                                                                                |

A PR editing both `projects/hipdnn/api/foo.h` and `dnn-providers/miopen-provider/src/bar.cpp` enters all six queues — core's row is a superset of every provider's row.

#### Membership at a glance

```
                                                       QUEUES
                    ┌────────┬──────────┬───────────┬────────────┬──────────┬─────────────┐
                    │        │  miopen  │ hipblaslt │ hip-kernel │ fusilli  │ integration │
PR touches          │ hipdnn │ provider │ provider  │  provider  │ provider │    tests    │
────────────────────┼────────┼──────────┼───────────┼────────────┼──────────┼─────────────┤
hipdnn core         │   ●    │    ●     │     ●     │     ●      │    ●     │      ●      │
miopen-provider     │   ·    │    ●     │     ·     │     ·      │    ·     │      ●      │
hipblaslt-provider  │   ·    │    ·     │     ●     │     ·      │    ·     │      ●      │
hip-kernel-provider │   ·    │    ·     │     ·     │     ●      │    ·     │      ●      │
fusilli-provider    │   ·    │    ·     │     ·     │     ·      │    ●     │      ●      │
integration-tests   │   ·    │    ·     │     ·     │     ·      │    ·     │      ●      │
────────────────────┴────────┴──────────┴───────────┴────────────┴──────────┴─────────────┘
   ● = PR enters this queue     · = PR does not enter this queue
```

Three patterns to read off the matrix:

- The `hipdnn core` row is the only fully-marked row — core is upstream of everything, so a core PR enters every downstream queue and is gated by every other in-flight PR in the ecosystem.
- Each provider row has exactly two marks — its own queue and its one downstream component, `integration-tests`. Providers serialize with each other only through the shared `integration-tests` queue.
- The `integration-tests` row has a single mark — it has no downstream components. Upstream PRs (providers, core) already enter the `integration-tests` queue, so serialization happens naturally.

#### Worked example: four PRs enqueued in order

PRs of four distinct types enter the queue in order — A (core), B (miopen-provider), C (hipblaslt-provider), D (integration-tests). Heads shown leftmost; `→` is "behind".

```
T₀ — all four enqueued

  hipdnn              │ A
  miopen-provider     │ A → B
  hipblaslt-provider  │ A      → C
  hip-kernel-provider │ A
  fusilli-provider    │ A
  integration-tests   │ A → B  → C  → D

  A is at the head of every queue → A merges next.

T₁ — after A merges

  miopen-provider     │ B
  hipblaslt-provider  │ C
  integration-tests   │ B → C → D

  B is at the head of {miopen-provider, integration-tests} → B merges next.
  C is at the head of hipblaslt-provider but blocked by B in integration-tests.
  D is blocked behind B and C in integration-tests.

T₂ — after B merges

  hipblaslt-provider  │ C
  integration-tests   │ C → D

  C is at the head of {hipblaslt-provider, integration-tests} → C merges next.
  D is still blocked by C in integration-tests.

T₃ — after C merges

  integration-tests   │ D

  D is at the head of integration-tests (its only queue) → D merges.
```

Two takeaways:

- **B and C never run in parallel** even though they touch different providers — they share `integration-tests`, so the integration suite serializes them. No cross-entry into each other's provider queues needed; the shared queue is sufficient.
- **If a second core PR E were enqueued at T₂**, it would join the tail of *every* queue — including `integration-tests` where C and D still sit — and would have to wait for both to clear before merging. Core is never allowed to overtake an in-flight provider or integration-tests PR.


### 4.3 Opt-in

Dependencies between components are expressed through queue membership in the `PATH_TO_QUEUES` config. The rule: **a PR enters its own queue plus the queue of every downstream component** — those that depend on it. When a component's PR sits in a downstream queue, it serializes with that component's PRs — that is the blocking relationship.

To opt in a new component, two edits are needed:

1. **Add your path entry.** Map your path prefix to a list of queues: your own queue, plus the queue of every component downstream of you. For example, a new provider whose only downstream component is `integration-tests`:
   ```python
   "dnn-providers/new-provider/": ["new-provider", "integration-tests"],
   ```
2. **Update entries of components upstream of you.** Any component that lists you as downstream must add your queue to its list. For hipDNN core (upstream of all providers), add `"new-provider"` to the `projects/hipdnn/` entry so that core PRs serialize with the new provider.

There is no per-project file. Removing the entry (and your queue from other entries) opts a project out cleanly. The initial opt-in set is hipDNN core + four providers + integration-tests. Other rocm-libraries projects are explicitly out of scope for v1.

### 4.4 PR lifecycle

1. An authorized user (PR author or any user with write/maintain/admin on the repo, per [§4.6](#46-permissions)) comments `/merge`.
2. The command handler validates eligibility (see [§4.6](#46-permissions)), computes the queue set from the PR's changed paths, applies a `mq:queued` label and one `mq:<queue>` label per queue, and posts a single status comment with a hidden JSON metadata marker.
3. A processor runs every 3 minutes. For each queue, it picks the head PR (oldest `enqueued_at`). If a PR is at the head of *every* queue it belongs to, the processor labels it `mq:active`, merges `develop` into the PR branch, and waits one cycle for CI to run against the freshly-merged tip.
4. On the next cycle:
   - **CI green** → squash-merge.
   - **CI still pending** → PR keeps `mq:active`, stays at the head, and is retried each cycle until checks settle.
   - **CI red** → eject with a comment naming the failure; the author re-enqueues with `/merge` after fixing.
   - **New commits pushed by a non-bot user while queued** → eject (the queue's "what we tested" guarantee no longer holds).
5. `/dequeue` removes a PR. The PR author or any write-access user can dequeue.

Labels:

- `mq:queued` — waiting at some position in one or more queues.
- `mq:active` — at the head of all its queues; CI cycle in progress.
- `mq:<queue>` — membership marker, one per queue the PR belongs to.

### 4.5 Cadence and concurrency

- 3-minute processor cron. Single concurrency group; no overlap, no cancellation.
- One PR processed per queue per cycle. No batching.
- The processor commits as `github-actions[bot]` so its merge of `develop` into the PR branch does not trigger the new-commit ejector in step 4.

### 4.6 Permissions

- `/merge`: PR author **or** any user with write/maintain/admin on the repo. The PR-author allowance is what makes the queue usable for external contributors on their own PRs (see [§5](#5-open-source-contributor-policy)).
- `/dequeue`: PR author or write-access user.
- The squash-merge itself is performed via GitHub's merge API by the bot, so branch-protection rules apply (see [§5](#5-open-source-contributor-policy)).

### 4.7 Processing model

Queue state is stored entirely in PR metadata — labels and a hidden JSON comment posted by the command handler at enqueue time — with no external database. The processor reconstructs the full queue picture on every cycle by scanning all open PRs with `mq:queued` or `mq:active` labels.

**Per-cycle algorithm.**

```
process_cycle():
    # 1. Discover: rebuild all queue state from PR metadata.
    pulls   = list_open_prs(base="develop", labels_any=["mq:queued", "mq:active"])
    members = {q: [] for q in ALL_QUEUES}
    for pr in pulls:
        meta = parse_metadata_comment(pr)            # enqueued_at, queues
        for q in meta.queues:
            members[q].append((meta.enqueued_at, pr))

    # 2. Build queues: each queue is a FIFO sorted by enqueue time.
    for q in ALL_QUEUES:
        members[q].sort(key=lambda entry: entry.enqueued_at)

    # 3. Identify ready PRs: head of every queue they belong to.
    ready = set()
    for pr in pulls:
        if all(members[q][0].pr == pr for q in pr.meta.queues):
            ready.add(pr)

    # 4. For each ready PR, advance its state machine.
    for pr in ready:
        if "mq:active" not in pr.labels:
            # Activate: rebase against develop, then wait one cycle for CI.
            result = merge_develop_into(pr.branch, author=BOT_USER)
            if result.is_conflict:
                eject(pr, reason="merge conflict with develop")
                continue
            set_labels(pr, add="mq:active", remove="mq:queued")
            continue                                  # CI must run before evaluation

        # Already active → evaluate the post-rebase CI cycle.
        if pushed_by_non_bot_since_activation(pr):
            eject(pr, reason="branch updated by author since activation")
        elif all_required_checks_passed(pr):
            squash_merge(pr)                          # GitHub merge API enforces protection
        elif any_required_check_failed(pr):
            eject(pr, reason=failed_check_name(pr))
        else:
            pass                                      # checks still pending; retry next cycle
```

A few invariants worth calling out:

- **No persistent processor state.** Every cycle starts from `list_open_prs(...)` and rebuilds `members` from scratch. If a cycle crashes mid-flight, the next cycle reconstructs the same picture from labels and metadata comments. State lives in the PR, not in the runner.
- **Activation and evaluation never happen in the same cycle.** A freshly activated PR always falls through to step 5 on the *next* cycle, which means CI sees the rebased tip before the queue acts on its result.
- **Per-queue FIFO is enforced by sort, not by a stored cursor.** The head of each queue is always the open PR with the oldest `enqueued_at` in that queue's member list — no separate "current head" pointer to fall out of sync.

**Consistency guarantees.** The single-concurrency-group constraint ([§4.5](#45-cadence-and-concurrency)) ensures no two processor runs overlap. Because state is reconstructed from PR metadata on every cycle, the processor is stateless and crash-safe — a failed run simply retries on the next 3-minute tick with no stale in-memory state.

**Conflict handling.** If merging `develop` into an active PR's branch produces a merge conflict, the PR is ejected immediately. The eject comment instructs the author to resolve conflicts locally, push, and re-enqueue with `/merge`.

## 5. Open-Source Contributor Policy

**Policy.** External contributors may use `/merge` on their own PRs. The safeguards that apply to any merge — required reviewer approvals and required CI checks — apply unchanged.

**Why this works without extra queue logic.** `develop` today routes PRs through a path-scoped CODEOWNERS reviewer set ([`.github/CODEOWNERS`](https://github.com/ROCm/rocm-libraries/blob/develop/.github/CODEOWNERS) — for example `/projects/hipdnn/` → `@ROCm/hipdnn-reviewers`) and is gated on the `Azure CI Summary` status check, documented as "a gating requirement" that "blocks the PR from merging" if it fails ([`docs/continuous-integration.md`](https://github.com/ROCm/rocm-libraries/blob/develop/docs/continuous-integration.md), §"PR Workflow"). GitHub's merge API enforces these rules on every merge call — including the queue's. If the required reviewers haven't approved or `Azure CI Summary` hasn't passed, GitHub rejects the squash. The queue piggybacks on branch protection; it cannot bypass it. Branch protection is the safety net, the queue is the serializer. These protections apply from Phase 1 onward (see [§6](#6-rollout)); the queue inherits them, it does not relax them.

**Defense-in-depth.** The command handler should also reject `/merge` on a PR that has not yet been approved, so an unapproved PR doesn't sit in the queue burning processor cycles only to fail at the squash. This is a small check at enqueue time, not a replacement for branch protection. (See Phase 2 in [§6](#6-rollout).)

**Latency.** No priority lane in v1. Uniform 3-minute poll for everyone. Equal latency is simpler and avoids second-tier optics. A priority lane is listed in [§8](#8-future-work).

## 6. Rollout

| Phase | What                                                                                                                                                                                                                                  |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | Optional. hipDNN-ecosystem opt-in only. No branch-protection changes.                                                                                                                                                                 |
| 2     | Configure `develop` branch protection: required reviewer approvals + required CI checks. Enable the at-enqueue approval pre-check. Add a non-blocking status check that flags non-queue merges so reviewers can nudge contributors.   |
| 3     | Add a required status check `merge-queue/managed` on `develop`. The queue bot posts it on every PR — pending until queue squash-merge for PRs touching opted-in paths, immediate success ("skipped — no managed paths") for all others. Path scoping lives in the bot, so branch protection stays one rule and other subprojects feel no friction. Repository admins (gardeners) retain bypass via GitHub's standard admin-override; this is an expected escape hatch for emergencies, not a routine path. (See [§7](#7-risks) for related concerns.) |
| 4     | Extend opt-in to other rocm-libraries subprojects on request.                                                                                                                                                                         |

## 7. Risks

- **3-minute poll** is slow at low load. Acceptable; revisit if it pinches.
- **Head-of-line stalls.** Cross-queue blocking means a slow core PR at the front of every queue holds up everything else. Mitigated by `/dequeue` (see [§4.4](#44-pr-lifecycle)) and reviewer discipline on core PRs.
- **Maintenance debt.** Custom in-repo Python is a maintenance cost compared to a hosted service. Accepted; revisit if the scripts grow beyond a single maintainer's head.
- **Serialization throughput.** Each queue cycle runs the full PR-validation CI: build jobs are bounded at 30 minutes ([`therock-ci-linux.yml`](https://github.com/ROCm/rocm-libraries/blob/develop/.github/workflows/therock-ci-linux.yml), `timeout-minutes: 30`), and individual test-component jobs are bounded at 210 minutes / 3.5 hours ([`therock-test-component.yml`](https://github.com/ROCm/rocm-libraries/blob/develop/.github/workflows/therock-test-component.yml), `timeout-minutes: 210`), with sharded matrices spanning multiple GPU architectures (`gfx942`, `gfx90a`, `gfx1201`, `gfx1100`, `gfx1030`) and OS targets. Wall-clock end-to-end CI for a coupled change can comfortably exceed 6 hours. Each cycle also pays setup overhead — pulling `ghcr.io/rocm/therock_build_manylinux_x86_64`, restoring caches, fanning out the matrix — that is fixed per attempt and not amortizable across queue cycles. Because intersecting PRs (e.g. two provider PRs sharing `integration-tests`) must merge one at a time, each waiting for a full CI cycle, a queue of *n* intersecting PRs takes roughly *n × CI time* to drain. During high-activity periods this could become a significant bottleneck. Batch merging (see [§8](#8-future-work)) is the primary mitigation; until then, reviewers should be aware that queuing order matters and core PRs at the head will block the entire ecosystem for one full CI cycle.
- **Surprise auto-eject** when an author pushes new commits mid-queue. The eject comment must be explicit about why and how to re-enqueue.

## 8. Future Work

Batch merging (`batch_size > 1`), a priority lane for short-running PRs, a dashboard listing all queue contents, auto-rebase on conflicts, and extending opt-in beyond the hipDNN ecosystem.

## Appendix A: Prototype reference

A working prototype implementing a near-cousin of this design lives at <https://github.com/SamuelReeder/rocm-libraries/tree/develop/.github>. It is intentionally underspecified here — the design above is the source of truth, and the prototype will be reshaped to match it before adoption.
