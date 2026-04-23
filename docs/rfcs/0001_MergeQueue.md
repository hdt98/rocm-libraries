# Federated Merge Queues for rocm-libraries

- Contributors: Samuel Reeder
- **Status**: First draft

## Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Problem Statement](#2-problem-statement)
- [3. Goals, Non-Goals, and Prerequisites](#3-goals-non-goals-and-prerequisites)
- [4. Design](#4-design)
  - [4.1 Queues](#41-queues)
  - [4.2 Path → Queue mapping](#42-path--queue-mapping)
  - [4.3 Data model](#43-data-model)
  - [4.4 Permissions](#44-permissions)
  - [4.5 PR lifecycle](#45-pr-lifecycle)
  - [4.6 Per-cycle algorithm](#46-per-cycle-algorithm)
  - [4.7 Cadence and concurrency](#47-cadence-and-concurrency)
  - [4.8 Opt-in / extension](#48-opt-in--extension)
  - [4.9 Implementation constraints](#49-implementation-constraints)
- [5. Open-Source Contributor Policy](#5-open-source-contributor-policy)
- [6. Rollout](#6-rollout)
- [7. Validation](#7-validation)
- [8. Risks](#8-risks)
- [9. Future Work](#9-future-work)
- [Appendix A: Prototype reference](#appendix-a-prototype-reference)

## 1. Executive Summary

This RFC proposes a custom, in-repo merge queue for rocm-libraries that serializes merges across coupled components on `develop`. Each opted-in component gets its own FIFO queue. A PR enters the queue of every component its changes could affect, and merges only when it is at the head of all of them. Authors trigger the queue with a `/merge` PR comment; an in-repo workflow squash-merges PRs as they reach the front.

Initial scope is the hipDNN ecosystem: `hipdnn` core, four providers (`miopen-provider`, `hipblaslt-provider`, `hip-kernel-provider`, `fusilli-provider`), and `integration-tests`. The queue is opt-in — other rocm-libraries subprojects are unaffected — starts optional (no branch-protection changes), and is designed so any project can join later by adding centralized config entries.

## 2. Problem Statement

Multiple PRs may merge into rocm-libraries without their cumulative state ever being tested. Because nothing serializes these merges, PRs that touch coupled code frequently experience semantic collisions: they may pass CI in isolation, but immediately break develop when merged concurrently.

The hipDNN ecosystem is especially susceptible. The various hipDNN-related components in rocm-libraries can be thought of as a Directed Acyclic Graph (DAG). To guarantee merge safety, downstream components must strictly serialize behind upstream changes. Without an automated system that enforces this serialization safely by default, we risk constant mainline regressions while forcing contributors to manually orchestrate their merges.

We desire an automated system that serializes with concurrent queues, is cheap to opt into, and stays out of contributors' way.

## 3. Goals, Non-Goals, and Prerequisites

### Goals
- One FIFO queue per component: `hipdnn`, four providers, `integration-tests`.
- Cross-blocking that follows dependencies (see [§4.2](#42-path--queue-mapping)). A PR is blocked by every PR ahead of it in any queue it belongs to.
- Opt-in per project. Subprojects outside the hipDNN ecosystem are unaffected unless they opt in.
- `develop` is the only managed target branch.
- Start optional; provide a clear path to "required" via branch protection.
- External contributors can use the queue with the same safeguards as anyone else.

### Non-Goals
- Replacing existing CI workflows.
- Batched merges (`batch_size > 1`) — deferred.
- Release or feature branches — `develop` only.
- A priority lane for short-running PRs — uniform 3-minute poll for everyone in v1; deferred to [§9](#9-future-work).

### Prerequisites

One external precondition the queue depends on:

- **`develop` branch protection enforces required reviewer approvals and required CI checks.** Per-component reviewer ownership is already declared in [`.github/CODEOWNERS`](https://github.com/ROCm/rocm-libraries/blob/develop/.github/CODEOWNERS); the required check is `TheRock CI Summary` ([`.github/workflows/therock-ci.yml`](https://github.com/ROCm/rocm-libraries/blob/develop/.github/workflows/therock-ci.yml)). The queue serializes; it does not authorize or bypass — see [§5](#5-open-source-contributor-policy) for why this is sufficient even for external contributors.

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

**Membership rule.** A PR enters its own queue plus the queue of every downstream component — those that depend on it and could break if it changes. This serializes the PR with any in-flight work in components it could affect.

**Dependency graph.** All providers depend on `hipdnn` core; `integration-tests` depends on every provider:

```
               ┌── miopen-provider ─────┐
               ├── hipblaslt-provider ──┤
hipdnn (core) ─┤                        ├──► integration-tests
               ├── hip-kernel-provider ─┤
               └── fusilli-provider ────┘
```

Applying the membership rule to this DAG produces the membership matrix below. Paths outside this graph (other rocm-libraries projects) are not opted in and enter no queues.

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

Three patterns follow directly from the DAG:

- The `hipdnn core` row is fully marked — core is upstream of everything, so a core PR is gated by every other in-flight PR in the ecosystem.
- Each provider row has exactly two marks — its own queue and `integration-tests`. Providers serialize with each other only through the shared `integration-tests` queue.
- The `integration-tests` row has a single mark — it has no downstream components.

A PR editing both `projects/hipdnn/api/foo.h` and `dnn-providers/miopen-provider/src/bar.cpp` enters all six queues — core's row is a superset of every provider's row.

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

### 4.3 Data model

Queue state is stored entirely in PR metadata, with no external database:

- **Membership labels.** One `mq:<queue>` label per queue the PR belongs to. Set at enqueue time; immutable so long as the PR remains in the queue.
- **State labels.** Either `mq:queued` (waiting at some position in one or more queues) or `mq:active` (at the head of all its queues; CI cycle in progress).
- **Status comment.** A single comment posted by the command handler on `/merge` and edited in place by the processor on every state change — including the terminal eject and merge states. Carries a hidden JSON payload (read by the processor) and a human-readable status block (read by authors). A per-PR comment is the natural fit because all the state we need to persist is itself per-PR, and the same comment doubles as the user-facing UI.

The processor reconstructs the full queue picture on every cycle by issuing one label-filtered API search per queue. A PR labelled `mq:hipdnn` *and* `mq:miopen-provider` is returned by both searches and appears in both queues' member lists.

#### Status comment format

A hidden marker plus a JSON header carries the fields the processor needs across cycles — `enqueued_at` (FIFO sort key), `queues` (membership set used by the algorithm's headship check), and `active_sha` (the head SHA recorded at activation; absent until activation, used to detect mid-cycle author pushes):

```html
<!-- mq-status -->
<!-- mq-meta: {"schema":1,"enqueued_at":"2026-04-22T15:23:14Z","queues":["miopen-provider","integration-tests"],"active_sha":"a1b2c3d"} -->
```

Other state (queue position, CI status, eject reason) is *not* persisted in the JSON — it's recomputed each cycle from labels and the live queue picture, then rendered into the visible body. Example queued-state body for a miopen-provider PR:

```markdown
## ⏳ Queued for merge

Enqueued by @samuel-reeder at 2026-04-22 15:23 UTC.

This PR touches `dnn-providers/miopen-provider/**`, entering 2 queues:

| Queue               | Position | Blocked by   |
| ------------------- | -------- | ------------ |
| `miopen-provider`   | 2 of 3   | #4521        |
| `integration-tests` | 3 of 5   | #4521, #4523 |

Comment `/dequeue` to leave the queue. Processor runs every 3 minutes.

_Updated 2026-04-22 15:26 UTC._
```

The active, ejected, and merged states swap the heading and body — active shows required-check status; ejected names the failing check and asks the author to re-enqueue with `/merge`; merged shows the squash SHA. The processor only edits when something actually changed (state, position, or check status), keeping edit history meaningful.

### 4.4 Permissions

- `/merge`: PR author **or** any user with write/maintain/admin on the repo. The PR-author allowance is what makes the queue usable for external contributors on their own PRs (see [§5](#5-open-source-contributor-policy)).
- `/dequeue`: PR author or any write-access user.
- The squash-merge is performed via GitHub's merge API by `github-actions[bot]`. Branch-protection rules on `develop` apply unchanged (see [§5](#5-open-source-contributor-policy)).

### 4.5 PR lifecycle

A PR moves through three observable states. Every transition is captured in the diagram below:

```
              /merge
   (none) ──────────────► mq:queued
                             │
              at head of all queues
                             ▼
                         mq:active ─── CI green ────► squash-merge
                          │  ▲
                          │  └── CI pending (next cycle)
                          │
                          ├── CI red ────────────────► eject
                          ├── merge conflict ────────► eject
                          └── HEAD changed ──────────► eject
```

Two GitHub Actions workflows drive these transitions:

1. **Command handler** *(event-driven, `issue_comment`)*. On `/merge`, validates eligibility (see [§4.4](#44-permissions)), confirms maintainer-edits is enabled for fork PRs (see [§5](#5-open-source-contributor-policy)), computes the queue set from the PR's changed paths via the rule in [§4.2](#42-path--queue-mapping), applies `mq:queued` and one `mq:<queue>` label per queue, and posts the status comment (see [§4.3](#43-data-model)). On `/dequeue`, removes all `mq:*` labels. This is the only workflow that assigns membership labels. The handler is idempotent — if the PR already carries `mq:queued` or `mq:active`, a second `/merge` is acknowledged as a no-op so concurrent invocations don't post duplicate status comments.
2. **Processor** *(scheduled, every 3 minutes)*. Activates the head PR of each queue (`mq:queued` → `mq:active`, then merges `develop` into the PR branch and records the resulting head SHA in the status comment's JSON), and on the next cycle evaluates CI to either squash-merge or eject. The same cycle ejects if the PR's current head SHA no longer matches the recorded `active_sha` — i.e., the author pushed since the last cycle. Both squash-merge and eject also clear all `mq:*` labels, so terminal PRs disappear from the discovery search on the next cycle (parallel to `/dequeue`'s cleanup). Algorithm in [§4.6](#46-per-cycle-algorithm).

After ejection, the author addresses the reported cause and re-enqueues with `/merge`.

### 4.6 Per-cycle algorithm

```
# ALL_QUEUES is the fixed set of six queue names defined in §4.1.

process_cycle():
    # 1. Discover: rebuild each queue by searching for its membership label,
    #    scoped to open PRs (terminal PRs have their mq:* labels cleared on
    #    squash-merge or eject; is:open is defense-in-depth against any leak).
    #    A PR labeled mq:hipdnn AND mq:miopen-provider is returned by both
    #    searches and appears in both members lists.
    members = {q: [] for q in ALL_QUEUES}
    for q in ALL_QUEUES:
        for pr in search_prs(query=f"is:pr is:open label:mq:{q}"):
            pr.meta = parse_metadata_comment(pr)      # enqueued_at, queues, active_sha?
            members[q].append(pr)
    pulls = {pr for q in ALL_QUEUES for pr in members[q]}

    # 2. Build queues: each queue is a FIFO sorted by enqueue time.
    for q in ALL_QUEUES:
        members[q].sort(key=lambda p: p.meta.enqueued_at)

    # 3. Identify ready PRs: head of every queue they belong to.
    ready = {pr for pr in pulls
             if all(members[q][0] == pr for q in pr.meta.queues)}

    # 4. For each ready PR, advance its state machine.
    for pr in ready:
        if "mq:active" not in pr.labels:
            # 4a. Activate: rebase against develop, record the resulting SHA,
            #     then wait one cycle for CI.
            result = merge_develop_into(pr.branch, author=BOT_USER)
            if result.is_conflict:
                eject(pr, reason="merge conflict with develop")
                continue
            update_status_comment(pr, set_active_sha=result.head_sha)
            set_labels(pr, add="mq:active", remove="mq:queued")
            continue                                  # CI must run before evaluation

        # 4b. Already active → evaluate the post-rebase CI cycle.
        if pr.head.sha != pr.meta.active_sha:
            eject(pr, reason="branch updated since activation")
        elif all_required_checks_passed(pr):
            squash_merge(pr, sha=pr.meta.active_sha)  # API call fails if HEAD has moved
        elif any_required_check_failed(pr):
            eject(pr, reason=failed_check_name(pr))
        else:
            pass                                      # checks still pending; retry next cycle
```

A few invariants worth calling out:

- **No persistent processor state.** Every cycle starts from `search_prs(...)` and rebuilds `members` from scratch. If a cycle crashes mid-flight, the next cycle reconstructs the same picture from labels and the status comment's hidden JSON. State lives in the PR, not in the runner.
- **Activation and evaluation never happen in the same cycle.** Step 4a ends with `continue`, so a freshly activated PR is evaluated only by step 4b on the *next* cycle — CI sees the rebased tip before the queue acts on its result.
- **Per-queue FIFO is enforced by sort, not by a stored cursor.** The head of each queue is always the open PR with the oldest `enqueued_at` in that queue's member list — no separate "current head" pointer to fall out of sync.

Crash-safety follows from the single concurrency group ([§4.7](#47-cadence-and-concurrency)) plus per-step idempotency. Activation does three writes — the `develop`-merge, the `active_sha` update, and the `mq:queued` → `mq:active` flip — and each is individually idempotent: re-merging `develop` after a successful merge is a no-op (or fast-forward), the `active_sha` write overwrites, and the label flip is idempotent on each side. A crash between any two writes leaves the PR in a state where the next cycle treats it as not-yet-activated and re-runs activation, converging to the same end state.

### 4.7 Cadence and concurrency

- 3-minute processor cron. Single concurrency group; no overlap, no cancellation.
- One PR processed per queue per cycle. No batching.
- `active_sha` is recorded *after* the bot's `develop`-merge, so subsequent SHA checks compare against the post-merge tip and the bot's own commit doesn't trigger an eject.

### 4.8 Opt-in / extension

To opt in a new component, two edits to the centralized `PATH_TO_QUEUES` config:

1. **Add your path entry.** Map your path prefix to a list of queues: your own queue, plus the queue of every component downstream of you (the membership rule from [§4.2](#42-path--queue-mapping)). For example, a new provider whose only downstream component is `integration-tests`:
   ```python
   "dnn-providers/new-provider/": ["new-provider", "integration-tests"],
   ```
2. **Update entries of components upstream of you.** Any component that lists you as downstream must add your queue to its list. For hipDNN core (upstream of all providers), add `"new-provider"` to the `projects/hipdnn/` entry so that core PRs serialize with the new provider.

There is no per-project file. Removing the entry (and your queue from other entries) opts a project out cleanly. The initial opt-in set is hipDNN core + four providers + integration-tests; other rocm-libraries projects are explicitly out of scope for v1.

### 4.9 Implementation constraints

The state-transition logic is structured as a pure function over queue state, separated from the GitHub-API I/O layer so the algorithm can be exercised against synthetic snapshots without any GitHub interaction (enabling the testing in [§7](#7-validation)). All writes go through `GITHUB_TOKEN` with workflow-scoped permissions; no branch-protection bypass. File layout, language, and library choices follow the existing repo conventions and are not normative here.

## 5. Open-Source Contributor Policy

**Policy.** External contributors may use `/merge` on their own PRs. The safeguards that apply to any merge — required reviewer approvals and required CI checks — apply unchanged.

**Why this works without extra queue logic.** GitHub's branch protection rules apply to the merge API by default — bypass requires explicit per-actor configuration ([GitHub docs: protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)) which the queue bot is not granted. So `develop`'s required `TheRock CI Summary` check and CODEOWNER reviews (see prereqs in [§3](#3-goals-non-goals-and-prerequisites)) must pass before any squash-merge succeeds — regardless of who or what calls the API. Branch protection is the safety net; the queue is the serializer.

**Defense-in-depth.** The command handler should also reject `/merge` on a PR that has not yet been approved, so an unapproved PR doesn't sit in the queue burning processor cycles only to fail at the squash. This is a small check at enqueue time, not a replacement for branch protection. (See Phase 2 in [§6](#6-rollout).)

**Fork PRs.** The processor pushes the `develop`-merge to the PR branch, which for fork PRs requires the author to have **Allow edits by maintainers** enabled (the GitHub default; authors can disable it — see [GitHub docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork)). The handler enforces this at `/merge` time and rejects with a clear message if disabled. If the contributor cannot or will not enable maintainer-edits, a maintainer may manually merge the PR after deeming it safe — the queue is opt-in, not the only path.

## 6. Rollout

| Phase | What                                                                                                                                                                                                                                  |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | Optional. hipDNN-ecosystem opt-in only. No branch-protection changes. Prerequisite: per-commit and fork-dogfood validation in [§7](#7-validation) is clean.                                                                            |
| 2     | Add the `merge-queue/managed` status check as **non-blocking**. The queue bot posts it on every PR — pending until queue squash-merge for opted-in paths, immediate success ("skipped — no managed paths") otherwise. Reviewers use it to nudge contributors toward `/merge` before promotion. Also enable the at-enqueue approval pre-check so unapproved PRs don't burn processor cycles. (Branch protection on `develop` already requires reviewer approvals and CI checks; no changes there.) |
| 3     | Promote `merge-queue/managed` to a **required** status check on `develop`. Path scoping lives in the bot, so branch protection stays one rule and other subprojects feel no friction. Repository admins (gardeners) retain bypass via GitHub's standard admin-override; this is an expected escape hatch for emergencies, not a routine path. (See [§8](#8-risks) for related concerns.) |
| 4     | Extend opt-in to other rocm-libraries subprojects on request.                                                                                                                                                                         |

## 7. Validation

Two layers of testing — per-commit on the queue's source, and end-to-end on a fork repo before any production exposure.

### Per-commit (Python)

Because the state-transition logic is separated from GitHub I/O (see [§4.9](#49-implementation-constraints)), the algorithm can be exercised against synthetic queue snapshots directly — no GitHub mocks required. Per-commit tests on the queue's source files should cover, at minimum:

- **Pure logic** — path → queue mapping, FIFO sort, headship check, status-comment JSON parse/serialize. The worked example in [§4.2](#42-path--queue-mapping) is a natural test case.
- **Algorithm invariants** — no PR squash-merged unless at the head of every queue it belongs to; no PR squash-merged with stale `active_sha`; re-enqueue is idempotent; FIFO order is respected. The exact harness is an implementation choice; what matters is that these invariants are exercised.

### Fork dogfood

Before promoting to Phase 1, implement and validate against a fork of `rocm-libraries`. Synthetic PRs against the fork exercise the full handler + processor flow on real GitHub APIs, demonstrating the invariants above and edge cases that aren't visible to the algorithm in isolation:

- merge conflict during activation
- CI failure during evaluation
- simultaneous `/merge` on the same PR
- author push between activation and squash

Promote to Phase 1 when invariants and edge cases pass cleanly.

## 8. Risks

- **3-minute poll** is slow at low load. Acceptable; revisit if it pinches.
- **Head-of-line stalls.** Cross-queue blocking means a slow core PR at the front of every queue holds up everything else. Mitigated by `/dequeue` (see [§4.5](#45-pr-lifecycle)) and reviewer discipline on core PRs.
- **Maintenance debt.** Custom in-repo Python is a maintenance cost compared to a hosted service. Accepted; revisit if the scripts grow beyond a single maintainer's head.
- **Serialization throughput — wall-clock end-to-end CI for a coupled change can comfortably exceed 6 hours.** Three contributing factors:
  - *Per-job timeouts.* Build jobs are bounded at 30 minutes ([`therock-ci-linux.yml`](https://github.com/ROCm/rocm-libraries/blob/develop/.github/workflows/therock-ci-linux.yml), `timeout-minutes: 30`); individual test-component jobs are bounded at 210 minutes / 3.5 hours ([`therock-test-component.yml`](https://github.com/ROCm/rocm-libraries/blob/develop/.github/workflows/therock-test-component.yml), `timeout-minutes: 210`), with sharded matrices spanning multiple GPU architectures (`gfx942`, `gfx90a`, `gfx1201`, `gfx1100`, `gfx1030`) and OS targets.
  - *Per-cycle setup overhead.* Pulling `ghcr.io/rocm/therock_build_manylinux_x86_64`, restoring caches, and fanning out the matrix is fixed per attempt and not amortizable across queue cycles.
  - *n × CI drain time.* Intersecting PRs (e.g. two provider PRs sharing `integration-tests`) merge one at a time, each waiting for a full CI cycle; a queue of *n* intersecting PRs takes roughly *n × CI time* to drain.

  Batch merging (see [§9](#9-future-work)) is the primary mitigation; until then, reviewers should be aware that queuing order matters and core PRs at the head will block the entire ecosystem for one full CI cycle.

- **No CI retries — flakes are terminal.** The processor treats any failed required check as a real failure (`any_required_check_failed → eject`). On a heterogeneous GPU matrix, transient hardware/runner failures eject otherwise-good PRs and force a manual `/merge` re-enqueue, paying another full CI cycle. This compounds with the throughput risk above: every flake-induced eject is another *n × CI* penalty. Mitigations to consider before Phase 3: a bounded retry on eject (e.g. retry once before ejecting), or a `/merge --ignore-flake <check-name>` escape hatch.
- **Surprise auto-eject** when an author pushes new commits mid-queue. Detection happens on the next processor cycle (up to 3 minutes after the push), so feedback is not instant — the eject comment must be explicit about why and how to re-enqueue.

## 9. Future Work

Batch merging (`batch_size > 1`), a priority lane for short-running PRs, retry-on-flake heuristics, and extending opt-in beyond the hipDNN ecosystem.

## Appendix A: Prototype reference

A working prototype implementing a near-cousin of this design lives at <https://github.com/SamuelReeder/rocm-libraries/tree/develop/.github>. It is intentionally underspecified here — the design above is the source of truth, and the prototype will be reshaped to match it before adoption.
