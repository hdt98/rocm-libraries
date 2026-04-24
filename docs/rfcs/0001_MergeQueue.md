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
- [6. Validation](#6-validation)
- [7. Rollout](#7-rollout)
- [8. Risks](#8-risks)
- [Appendix A: Prototype reference](#appendix-a-prototype-reference)

## 1. Executive Summary

This RFC proposes a custom, in-repo **federated** merge queue for rocm-libraries on `develop`: per-component FIFO queues that interconnect along the dependency graph, rather than a single global queue serializing everything. A PR enters the queue of every component its changes could affect, and merges only when it is at the head of all of them. Authors trigger the queue with a `/merge` PR comment; an in-repo workflow squash-merges PRs as they reach the front.

Initial scope is the hipDNN ecosystem: `hipdnn` core, four providers (`miopen-provider`, `hipblaslt-provider`, `hip-kernel-provider`, `fusilli-provider`), and `integration-tests`. The queue is opt-in (other rocm-libraries subprojects are unaffected) and starts optional, with no branch-protection changes; any project can join later by adding centralized config entries.

## 2. Problem Statement

Multiple PRs may merge into rocm-libraries without their cumulative state ever being tested. Because nothing serializes these merges, PRs that touch coupled code frequently experience semantic collisions: they may pass CI in isolation, but immediately break develop when merged concurrently.

The hipDNN ecosystem is especially susceptible. The various hipDNN-related components in rocm-libraries can be thought of as a Directed Acyclic Graph (DAG). To guarantee merge safety, downstream components must strictly serialize behind upstream changes. Without an automated system that enforces this serialization safely by default, we risk constant mainline regressions while forcing contributors to manually orchestrate their merges.

We desire an automated system that serializes with concurrent queues, is cheap to opt into, and stays out of contributors' way.

## 3. Goals, Non-Goals, and Prerequisites

### Goals
- One FIFO queue per opted-in component. (Initial scope is the hipDNN ecosystem; see [§4.1](#41-queues).)
- Cross-blocking that follows dependencies (see [§4.2](#42-path--queue-mapping)). A PR is blocked by every PR ahead of it in any queue it belongs to.
- Opt-in per project. Subprojects outside the hipDNN ecosystem are unaffected unless they opt in.
- `develop` is the only managed target branch.
- Start optional; provide a clear path to "required" via branch protection.
- External contributors can use the queue with the same safeguards as anyone else.

### Non-Goals
- Replacing existing CI workflows.
- Batched merges (`batch_size > 1`).
- Release or feature branches — `develop` only.
- Priority labels to skip to the front of the queue — potentially future work.

### Prerequisites

One external precondition the queue depends on:

- **`develop` branch protection enforces required reviewer approvals and required CI checks.** Per-component reviewer ownership is already declared in [`.github/CODEOWNERS`](https://github.com/ROCm/rocm-libraries/blob/develop/.github/CODEOWNERS); the required check is `TheRock CI Summary` ([`.github/workflows/therock-ci.yml`](https://github.com/ROCm/rocm-libraries/blob/develop/.github/workflows/therock-ci.yml)). The queue serializes; it does not authorize or bypass — see [§5](#5-open-source-contributor-policy) for why this is sufficient even for external contributors.

## 4. Design

### 4.1 Queues

Each opted-in component gets one FIFO queue. The initial scope is the hipDNN ecosystem, with six queues:

- `hipdnn`
- `miopen-provider`
- `hipblaslt-provider`
- `hip-kernel-provider`
- `fusilli-provider`
- `integration-tests`

Other rocm-libraries subprojects opt in by adding entries to the centralized config (see [§4.8](#48-opt-in--extension)); they have no queue otherwise.

Each PR enters zero or more queues based on the paths it touches. A PR merges only when at the head of **every** queue it belongs to.

### 4.2 Path → Queue mapping

**Membership rule.** A PR enters its own queue plus the queue of every downstream component its changes could break.

**Dependency graph.** Providers depend on both `hipdnn` core (through hipDNN's plugin interface) and `integration-tests` (each provider's CI runs the integration suite). Integration-tests in turn depend on hipDNN's public headers:

```
hipdnn (core) ──► integration-tests ──► providers (×4)
       │                                       ▲
       └───────────────────────────────────────┘
```

The relationship between providers and integration-tests is asymmetric:

- *Provider → integration-tests is decoupled.* Provider changes can only affect integration-tests through hipDNN's stable plugin contract, so even coupled provider changes are unlikely to break the integration suite — though a contract violation could still cause a regression. Provider PRs therefore don't enter the integration-tests queue.
- *Integration-tests → providers is not decoupled.* A change to integration-tests directly changes what every provider's CI must satisfy. Integration-test PRs therefore enter every provider queue.

Paths outside this graph (other rocm-libraries projects) are not opted in and enter no queues.

#### Membership at a glance

```
                                                       QUEUES
                    ┌────────┬──────────┬───────────┬────────────┬──────────┬─────────────┐
                    │        │  miopen  │ hipblaslt │ hip-kernel │ fusilli  │ integration │
PR touches          │ hipdnn │ provider │ provider  │  provider  │ provider │    tests    │
────────────────────┼────────┼──────────┼───────────┼────────────┼──────────┼─────────────┤
hipdnn core         │   ●    │    ●     │     ●     │     ●      │    ●     │      ●      │
miopen-provider     │   ·    │    ●     │     ·     │     ·      │    ·     │      ·      │
hipblaslt-provider  │   ·    │    ·     │     ●     │     ·      │    ·     │      ·      │
hip-kernel-provider │   ·    │    ·     │     ·     │     ●      │    ·     │      ·      │
fusilli-provider    │   ·    │    ·     │     ·     │     ·      │    ●     │      ·      │
integration-tests   │   ·    │    ●     │     ●     │     ●      │    ●     │      ●      │
────────────────────┴────────┴──────────┴───────────┴────────────┴──────────┴─────────────┘
   ● = PR enters this queue     · = PR does not enter this queue
```

Three patterns follow:

- The `hipdnn core` row is fully marked — core sits upstream of every component and could break any of them.
- The `integration-tests` row is marked in every provider column — provider CI runs the integration suite, so an integration-tests change can break any provider. The `hipdnn` column is unmarked because `hipdnn` doesn't depend on integration-tests.
- Each provider row has a single mark — providers are leaves of the dependency graph, with nothing downstream of them. The plugin contract keeps integration-tests insulated from provider changes.

A PR editing both `projects/hipdnn/api/foo.h` and `dnn-providers/miopen-provider/src/bar.cpp` enters all six queues — core's row is a superset of every provider's row.

#### Worked example: five PRs across four time-steps

Four PRs are enqueued at T₀ — A (core), B (miopen-provider), C (hipblaslt-provider), D (integration-tests). A second core PR, E, is enqueued mid-flight at T₁ to demonstrate how core serializes behind in-flight downstream work. Heads shown leftmost; `→` is "behind".

```
T₀ — A, B, C, D enqueued

  hipdnn              │ A
  miopen-provider     │ A → B           → D
  hipblaslt-provider  │ A      → C      → D
  hip-kernel-provider │ A               → D
  fusilli-provider    │ A               → D
  integration-tests   │ A               → D

  A is at the head of every queue → A merges next.

T₁ — A merged; E (a second core PR) enqueued

  hipdnn              │ E
  miopen-provider     │ B → D → E
  hipblaslt-provider  │ C → D → E
  hip-kernel-provider │ D → E
  fusilli-provider    │ D → E
  integration-tests   │ D → E

  B and C each sit at the head of their only queue → they merge in parallel.
  D is at the head of {hip-kernel-provider, fusilli-provider, integration-tests}
  but blocked by B and C in the other two provider queues.
  E is alone at the head of hipdnn but blocked behind D in all 5 other queues.

T₂ — B and C merged

  hipdnn              │ E
  miopen-provider     │ D → E
  hipblaslt-provider  │ D → E
  hip-kernel-provider │ D → E
  fusilli-provider    │ D → E
  integration-tests   │ D → E

  D is at the head of every queue it belongs to → D merges.
  E remains blocked behind D in 5 of its 6 queues.

T₃ — D merged

  hipdnn              │ E
  miopen-provider     │ E
  hipblaslt-provider  │ E
  hip-kernel-provider │ E
  fusilli-provider    │ E
  integration-tests   │ E

  E is now at the head of every queue → E merges.
```

Two takeaways:

- **B and C merge in parallel; D is held up behind them.** Distinct provider PRs share no queues with each other (no double-serialization through integration-tests), but an integration-tests PR crosses every provider queue. The integration suite is shared infrastructure that providers depend on, so changes to it must serialize against any in-flight provider work.
- **E (core) joins the tail of every queue at T₁ and merges last, after B/C/D have all cleared.** Core is never allowed to overtake an in-flight downstream PR — even though E sits alone at the head of `hipdnn` from T₁ onward, it has to wait through three full CI cycles (B/C, then D, then itself) before merging.

### 4.3 Data model

Queue state lives in the following PR-attached primitives, each chosen so that its mutability is either constrained by GitHub's own access controls or explicitly accounted for by the tamper-resistance design (see [§4.3.1](#431-tamper-resistance)):

- **Membership labels.** One `mq:<queue>` label per queue the PR belongs to. Set at enqueue time; immutable so long as the PR remains in the queue.
- **State labels.** Either `mq:queued` (waiting at some position in one or more queues) or `mq:active` (at the head of all its queues; CI cycle in progress).
- **Issue timeline.** GitHub records every label application/removal as a timeline event with the actor's identity and timestamp. The timeline is append-only and not editable through any API. The processor reads it for two derived facts: the FIFO sort key (timestamp of the most recent `github-actions[bot]`-applied `mq:queued` event) and the actor of every `mq:*` label change (used to detect external tampering).
- **Activation marker.** A `merge-queue/active` commit status, created by the bot on the head SHA when the PR transitions to `mq:active`. The binding check is "does the *current* head SHA carry a `merge-queue/active` status whose `creator.login` is `github-actions[bot]`?" — a force-push or new commit produces a new head SHA which lacks the bot's status, naturally invalidating the activation. The `creator` filter matters because GitHub does not scope status creation by context: anyone with `statuses: write` can post a status with the same name. The lookup ignores statuses from other principals, so a maintainer or external app posting a competing `merge-queue/active` status under their own identity does not satisfy the activation check. (Distinct from the future `merge-queue/managed` required check in [§4.5](#45-pr-lifecycle); `managed` enforces queue routing on `develop`, `active` binds activation to a specific commit.)
- **Status comment.** A single bot-posted comment that renders the human-readable status block. A hidden `<!-- mq-status -->` marker lets the processor find and edit its own comment in place rather than posting a fresh one on every state change. The comment is a UI surface only; the processor reads no state from it.

The processor reconstructs the full queue picture on every cycle by issuing one label-filtered API search per queue. A PR labelled `mq:hipdnn` *and* `mq:miopen-provider` is returned by both searches and appears in both queues' member lists.

#### Status comment format

The comment carries the marker and a human-readable body:

```html
<!-- mq-status -->
```

Derived state (queue position, CI status, eject reason) is recomputed each cycle from labels + the live queue picture and rendered into the visible body. Example queued-state body for a miopen-provider PR:

```markdown
## ⏳ Queued for merge

Enqueued by @samuel-reeder at 2026-04-22 15:23 UTC.

This PR touches `dnn-providers/miopen-provider/**`, entering 1 queue:

| Queue             | Position | Blocked by |
| ----------------- | -------- | ---------- |
| `miopen-provider` | 2 of 3   | #4521      |

Comment `/dequeue` to leave the queue. Processor runs every 3 minutes.

_Updated 2026-04-22 15:26 UTC._
```

The active, ejected, and merged states swap the heading and body — active shows required-check status; ejected names the failing check and asks the author to re-enqueue with `/merge`; merged shows the squash SHA. The processor only edits when something actually changed (state, position, or check status), keeping edit history meaningful.

### 4.3.1 Tamper resistance

Labels are the only canonical state surface mutable by non-bot actors (any user with `triage` repo permission can add or remove them). The issue timeline is append-only and not editable through any API; the `merge-queue/active` commit status is scoped by GitHub to the creating principal, so other apps and users cannot overwrite it; and the status comment is not authoritative — it's a UI surface the bot regenerates as needed. Every tampering vector maps to one of three outcomes:

- *Cannot tamper* — GitHub access controls prevent the action (commit-status overwrites by other principals).
- *No-op* — the surface is non-authoritative; the bot regenerates or re-posts as needed (any edit, deletion, or duplication of the status comment).
- *Detect → eject* — the bot observes the tamper and clears the PR from all queues with an explanatory comment (any external `mq:*` label change).

No tamper produces an unsafe merge.

This protects against (a) accidental edits by repo maintainers, (b) other automation that fires on `mq:*` labels by mistake, and (c) a single non-bot account misusing repo permissions. It does *not* defend against a compromised workflow runner — at that point the bot's auth is the attacker's auth, and the workflow itself can do anything the queue can (acknowledged in [§8](#8-risks)).

#### Tamper matrix

| Tamper vector                                               | Who can do it                  | Response                                                                                                       |
| ----------------------------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Add any `mq:*` label                                        | Triage+                       | **Eject.** Real-time audit fires; clears all `mq:*` labels, removes activation status, posts comment.          |
| Remove any `mq:*` label                                     | Triage+                       | **Eject.** Same path — clear remaining `mq:*` labels, post comment.                                            |
| Edit, delete, or plant a duplicate of the status comment    | Comment author / Maintain+ / Sibling workflow | **No-op for correctness; possible UI degradation.** The comment is a UI surface, so no merge decision is affected. If the bot's marker-lookup picks a non-bot comment (sibling workflow planting `<!-- mq-status -->`, or a maintainer crafting one), status updates may land on the wrong comment until labels clear. Accepted as UI-only degradation. |
| Plant a competing `merge-queue/active` status               | Anyone with `statuses: write` | **No-op.** The bot's lookup filters by `creator.login == github-actions[bot]`, so a status created by a maintainer's PAT or an external app is ignored. (A *sibling workflow* posting via `GITHUB_TOKEN` shares the bot's identity — covered by the workflow-trust boundary in [§8](#8-risks).) |
| Force-push / new commit on the PR branch                    | PR author                     | **Eject (next cycle).** New head SHA has no `merge-queue/active` status — covered in [§4.5](#45-pr-lifecycle). |
| Manually invoke `/merge` without eligibility                | Anyone who can comment        | **Rejected at handler.** Eligibility check in [§4.4](#44-permissions).                                         |
| Concurrent `/merge` from multiple commenters                | Anyone                        | **Idempotent.** Handler short-circuits if PR already carries `mq:queued` or `mq:active`.                       |
| Spoof the bot identity (compromised token / app key)        | Workflow runner               | **Out of scope.** Compromised runner has the bot's authority; documented as a [§8](#8-risks) risk.             |

The matrix walks single-action tampers; combined sequences within a single window (e.g., a triage user removing one `mq:*` label and re-adding another in quick succession) are not separately enumerated and rely on the eject-on-detection path being hit by *any* one of the constituent actions.

#### Real-time tamper audit

The audit is implemented as a second job inside `mq-handler.yml` — the same workflow that handles `/merge` and `/dequeue`. Collapsing both into one file keeps every event-reactive queue-state mutator in one place. The handler workflow declares two triggers: `issue_comment: [created]` for commands, and `pull_request_target: [labeled, unlabeled, opened, reopened]` for the audit. Jobs dispatch on `github.event_name`. The audit job is gated to events whose `label.name` starts with `mq:`. If `event.sender.login == "github-actions[bot]"` (and `user.type == "Bot"` — see identity-check note below), the job is a no-op. Otherwise it clears all `mq:*` labels on the PR, removes the `merge-queue/active` commit status if present, and posts a comment naming the actor and explaining the eject. The `opened`/`reopened` triggers catch PRs created with `mq:*` labels pre-applied (e.g., copied from a template or branched from a PR fork).

**Why `pull_request_target` is safe here.** The standard `pull_request_target` risk (running untrusted PR code with the upstream's full token) requires checking out PR code, which the audit never does — it reads only event-payload fields and calls GitHub APIs. This is the same pattern the existing `.github/workflows/labeler.yml` uses for fork-PR labelling. The audit's trust scope matches the cron processor, which already exercises full upstream authority on fork PRs (squash-merge, `develop`-merge, label and status writes), so this adds no new privilege boundary. Required hygiene for `mq-handler.yml`: pass event-payload strings via env vars rather than `${{ }}`-interpolating into shell; declare a minimal `permissions:` block (`pull-requests: write`, `issues: write`, `statuses: write`, `contents: read`); set `concurrency: mq-handler-${{ github.event.pull_request.number }}` to serialize per-PR; and treat the workflow file itself as security-sensitive under `.github/` branch protection and CODEOWNERS.

**Audit's role for label removals specifically.** When a user removes `mq:queued` from a PR, the PR no longer carries the label, so the processor's discovery (`search_prs(label:mq:<queue>)`) doesn't return it on the next cycle — the cycle-time backstop never sees it. The real-time `unlabeled` event is the *only* mechanism for catching a removal. For label *additions*, the cycle-time backstop is sufficient (the new label brings the PR into discovery scope, the actor check then ejects), so real-time response on `labeled` is a latency optimization rather than the only line of defense.

#### Cycle-time backstop

If the audit workflow misfires or is disabled, the processor's discovery loop independently verifies, for every PR returned by `search_prs(label:mq:<queue>)`:

1. The most recent `mq:*` label-application event was applied by `github-actions[bot]` (timeline lookup).
2. For PRs labelled `mq:active`, the current head SHA carries the `merge-queue/active` commit status.

Any failure routes through the same eject path as the audit workflow.

#### Identity-check robustness

The "actor is the bot" check must not rely on a `login` string compare alone. It additionally verifies `user.type == "Bot"` (and ideally pins the App slug) so a coincidentally-named user account or a different installed App with similar attribution cannot impersonate the bot identity. Separately, `triage` permission is broader than label management: it also lets a user dismiss reviews and re-request changes. The [§4.4](#44-permissions) eligibility check for `/merge` therefore cannot trust the `author_association` field from the issue-comment event payload (which may be stale or report `CONTRIBUTOR`/`NONE` for users who do hold write access); it must call `repos.getCollaboratorPermissionLevel` live to verify write-access at the moment the command is issued.

`mq:*` is a reserved label namespace; CONTRIBUTING will document that maintainers should not apply or remove these labels by hand.

### 4.4 Permissions

- `/merge`: PR author **or** any user with write/maintain/admin on the repo. The PR-author allowance is what makes the queue usable for external contributors on their own PRs (see [§5](#5-open-source-contributor-policy)). The handler verifies write-access by calling `repos.getCollaboratorPermissionLevel` live at the moment the command is issued — not by trusting `author_association` from the issue-comment payload, which can be stale or report `CONTRIBUTOR`/`NONE` for users who do hold write access (see [§4.3.1](#431-tamper-resistance)).
- `/dequeue`: PR author or any write-access user (verified the same way).
- The squash-merge is performed via GitHub's merge API by `github-actions[bot]`. Branch-protection rules on `develop` apply unchanged (see [§5](#5-open-source-contributor-policy)).

### 4.5 PR lifecycle

A PR moves through three observable states. Every transition is captured in the diagram below:

```
              /merge
   (none) ──────────────► mq:queued ─── external mq:* edit ──► eject
                             │
              at head of all queues
                             ▼
                         mq:active ─── CI green ────► squash-merge
                          │  ▲
                          │  └── CI pending (next cycle)
                          │
                          ├── CI red ────────────────► eject
                          ├── merge conflict ────────► eject
                          ├── HEAD changed ──────────► eject
                          └── external mq:* edit ────► eject
```

Two GitHub Actions workflows drive these transitions:

1. **Command handler + tamper audit** *(event-driven, `mq-handler.yml`)*. Two jobs in one workflow file, dispatched on `github.event_name`:
   - On `issue_comment: [created]` matching `/merge`: validates eligibility (see [§4.4](#44-permissions)), checks the PR has at least one approving review, confirms maintainer-edits is enabled for fork PRs (see [§5](#5-open-source-contributor-policy)), computes the queue set from the PR's changed paths via the rule in [§4.2](#42-path--queue-mapping), applies `mq:queued` and one `mq:<queue>` label per queue, and posts the status comment (see [§4.3](#43-data-model)). On `/dequeue`, removes all `mq:*` labels. This is the only workflow that assigns membership labels. The handler is idempotent — if the PR already carries `mq:queued` or `mq:active`, a second `/merge` is acknowledged as a no-op so concurrent invocations don't post duplicate status comments.
   - On `pull_request_target: [labeled, unlabeled, opened, reopened]` for `mq:*` labels: runs the real-time tamper audit (see [§4.3.1](#431-tamper-resistance)) — eject if the actor is not `github-actions[bot]`.
2. **Processor** *(scheduled, every 3 minutes, `mq-processor.yml`)*. Activates the head PR of each queue (`mq:queued` → `mq:active`, then merges `develop` into the PR branch and creates a `merge-queue/active` commit status on the resulting head SHA — see [§4.3](#43-data-model)), and on the next cycle evaluates CI to either squash-merge or eject. The cycle's headship check verifies that the PR's current head SHA still carries the `merge-queue/active` status; if a new commit was pushed in between, the new head SHA lacks the status and the PR is ejected. The same cycle also runs the tamper-resistance backstop checks from [§4.3.1](#431-tamper-resistance). Both squash-merge and eject clear all `mq:*` labels and remove the activation status, so terminal PRs disappear from the discovery search on the next cycle (parallel to `/dequeue`'s cleanup). Algorithm in [§4.6](#46-per-cycle-algorithm).

After ejection, the author addresses the reported cause and re-enqueues with `/merge`.

#### Branch protection integration

Once the queue has matured through its initial rollout (see [§7](#7-rollout)), a `merge-queue/managed` GitHub status check will serve as the enforcement path. Initially non-blocking, it can be promoted to a required check on `develop` to make the queue the enforced merge path for opted-in components without affecting other subprojects. The design of this check is future work.

### 4.6 Per-cycle algorithm

*The following is functional pseudocode illustrating the algorithm's logic and sequencing. It is not intended as a literal implementation.*

```
# ALL_QUEUES is the set of opted-in queue names from §4.1 (six in the initial scope).

process_cycle():
    # 1. Discover: rebuild each queue by searching for its membership label,
    #    scoped to open PRs (terminal PRs have their mq:* labels cleared on
    #    squash-merge or eject; is:open is defense-in-depth against any leak).
    #    A PR labeled mq:hipdnn AND mq:miopen-provider is returned by both
    #    searches and appears in both members lists.
    members = {q: [] for q in ALL_QUEUES}
    for q in ALL_QUEUES:
        for pr in search_prs(query=f"is:pr is:open label:mq:{q}"):
            members[q].append(pr)
    pulls = {pr for q in ALL_QUEUES for pr in members[q]}

    # 2. Tamper-resistance backstop (see §4.3.1) + canonical-state derivation.
    #    For each candidate PR: if the issue timeline shows the most recent
    #    mq:* label change came from a non-bot actor, eject the PR and exclude
    #    it from this cycle. For surviving PRs, derive:
    #      - enqueued_at         — timestamp of the bot's most recent mq:queued
    #                              label-application event (from the timeline)
    #      - queues              — mq:<queue> labels minus the state labels
    #      - is_validly_active   — mq:active label AND a merge-queue/active
    #                              commit status created by github-actions[bot]
    #                              present on the current head SHA
    #    is_validly_active is consumed by step 5b's eject path, not used here.
    pulls -= eject_externally_modified(pulls, members)
    for pr in pulls:
        derive_canonical_state(pr)

    # 3. Build queues: each queue is a FIFO sorted by enqueue time.
    for q in ALL_QUEUES:
        members[q].sort(key=lambda p: p.enqueued_at)

    # 4. Identify ready PRs: head of every queue they belong to.
    ready = {pr for pr in pulls
             if all(members[q][0] == pr for q in pr.queues)}

    # 5. For each ready PR, advance its state machine.
    for pr in ready:
        if "mq:active" not in pr.labels:
            # 5a. Activate: merge develop into the branch, create the
            #     merge-queue/active commit status on the resulting SHA,
            #     then wait one cycle for CI.
            result = merge_develop_into(pr.branch, author=BOT_USER)
            if result.is_conflict:
                eject(pr, reason="merge conflict with develop")
                continue
            mark_active(result.head_sha)              # creates merge-queue/active commit status
            set_labels(pr, add="mq:active", remove="mq:queued")
            update_status_comment(pr)                 # rewrite visible body for new state
            continue                                  # CI must run before evaluation

        # 5b. Already active → evaluate the post-merge CI cycle.
        if not pr.is_validly_active:
            # Either branch was updated since activation (new head SHA lacks the
            # status), or mq:active was applied externally without an activation
            # status — both are unsafe to merge.
            eject(pr, reason="activation invalid (branch updated or label tampered)")
        elif all_required_checks_passed(pr):
            squash_merge(pr, sha=pr.head.sha)         # API call fails if HEAD has moved
        elif any_required_check_failed(pr):
            eject(pr, reason=failed_check_name(pr))
        else:
            pass                                      # checks still pending; retry next cycle
```

A few invariants worth calling out:

- **No persistent processor state.** Every cycle starts from `search_prs(...)` and rebuilds `members` from scratch. If a cycle crashes mid-flight, the next cycle reconstructs the same picture from canonical sources (labels, timeline, commit status). State lives in the PR, not in the runner.
- **Tamper checks run before any state-mutating step.** Step 2 ejects any PR whose canonical state is inconsistent with bot-only mutation — see [§4.3.1](#431-tamper-resistance). The eject path it uses is the same one used by the real-time audit job, so the two surfaces converge.
- **Activation and evaluation never happen in the same cycle.** Step 5a ends with `continue`, so a freshly activated PR is evaluated only by step 5b on the *next* cycle — CI sees the post-merge tip before the queue acts on its result.
- **Per-queue FIFO is enforced by sort, not by a stored cursor.** The head of each queue is always the open PR with the oldest `enqueued_at` in that queue's member list — no separate "current head" pointer to fall out of sync.

Crash-safety follows from the single concurrency group ([§4.7](#47-cadence-and-concurrency)) plus per-step idempotency. Activation does three writes — the `develop`-merge, the `merge-queue/active` status creation, and the `mq:queued` → `mq:active` label flip — and each is individually idempotent: re-merging `develop` after a successful merge is a no-op (or fast-forward), creating a commit status on the same SHA with the same context overwrites, and the label flip is idempotent on each side. A crash between any two writes leaves the PR in a state where the next cycle treats it as not-yet-activated and re-runs activation, converging to the same end state.

### 4.7 Cadence and concurrency

- 3-minute processor cron. Single concurrency group; no overlap, no cancellation.
- One PR processed per queue per cycle. No batching.
- The `merge-queue/active` commit status is created *on* the SHA produced by the bot's `develop`-merge, so subsequent cycles' presence checks compare against the post-merge tip and the bot's own merge commit doesn't trigger an eject.

### 4.8 Opt-in / extension

Any component (in or outside the hipDNN ecosystem) opts in by editing the centralized `PATH_TO_QUEUES` config:

1. **Add a path entry for your component.** Map your path prefix to a list of queues: your own queue, plus the queue of every downstream component your changes could break (the membership rule from [§4.2](#42-path--queue-mapping)). A *leaf* component (nothing downstream) lists only its own queue; an upstream component lists every downstream queue too.
2. **Update entries of components upstream of you.** Any existing entry that names a component you're downstream of must add your queue to its list, so upstream PRs serialize with you.

Concrete example — opting in a new provider in the hipDNN ecosystem:

1. Add the new provider's path entry. Providers are leaves of the dependency graph (nothing downstream of them), so it enters only its own queue:
   ```python
   "dnn-providers/new-provider/": ["new-provider"],
   ```
2. Add `"new-provider"` to the `projects/hipdnn/` and `dnn-providers/integration-tests/` entries — both are upstream of every provider, so their PRs need to serialize against the new one.

There is no per-project file. Removing the entry (and your queue from other entries) opts a project out cleanly. The initial opt-in set is hipDNN core + four providers + integration-tests; other rocm-libraries projects are explicitly out of scope for v1.

### 4.9 Implementation constraints

Each processor cycle will follow a read → decide → execute structure. First, queue state is fetched from GitHub (PR labels, enqueue timestamps, head SHAs, CI status). That snapshot is passed to a pure decision function that returns the set of actions to take — squash, activate, eject, wait — without performing any I/O itself. Finally, those actions are dispatched back to GitHub. Keeping the decision layer pure means tests can construct any queue state as plain data and call the function directly, with no mocks or network access required. All writes go through `GITHUB_TOKEN` with workflow-scoped permissions.

## 5. Open-Source Contributor Policy

**Policy.** External contributors may use `/merge` on their own PRs. The safeguards that apply to any merge — required reviewer approvals and required CI checks — apply unchanged.

**Why this works without extra queue logic.** GitHub's branch protection rules apply to the merge API by default — bypass requires explicit per-actor configuration ([GitHub docs: protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)). If the bot account is granted bypass rights, the implementation must check that all required statuses have passed before squash-merging; otherwise branch protection enforces this automatically. `develop`'s required `TheRock CI Summary` check and CODEOWNER reviews (see prereqs in [§3](#3-goals-non-goals-and-prerequisites)) serve as the gate.

**Defense-in-depth.** The command handler must reject `/merge` on a PR that has not yet been approved, so an unapproved PR doesn't sit in the queue burning processor cycles only to fail at the squash. This is a small check at enqueue time, not a replacement for branch protection. Enabled from Phase 1 of [§7](#7-rollout).

**Fork PRs.** The processor pushes the `develop`-merge to the PR branch, which for fork PRs requires the author to have **Allow edits by maintainers** enabled (the GitHub default; authors can disable it — see [GitHub docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork)). The handler enforces this at `/merge` time and rejects with a clear message if disabled. If the contributor cannot or will not enable maintainer-edits, a maintainer may manually merge the PR after deeming it safe — the queue is opt-in, not the only path.

## 6. Validation

Two layers of testing — per-commit on the queue's source, and end-to-end on a fork repo before any production exposure.

### Per-commit (Python)

Because the state-transition logic is separated from GitHub I/O (see [§4.9](#49-implementation-constraints)), the algorithm can be exercised against synthetic queue snapshots directly — no GitHub mocks required. Per-commit tests on the queue's source files should cover, at minimum:

- **Pure logic** — path → queue mapping, FIFO sort, headship check, derivation of `enqueued_at`/`queues`/`is_validly_active` from canonical sources (see [§4.3](#43-data-model), [§4.6](#46-per-cycle-algorithm)). The worked example in [§4.2](#42-path--queue-mapping) is a natural test case.
- **Algorithm invariants** — no PR squash-merged unless at the head of every queue it belongs to; no PR squash-merged whose head SHA lacks the `merge-queue/active` commit status; re-enqueue is idempotent; FIFO order is respected; PRs whose `mq:*` labels were last modified by a non-bot actor are ejected before any state-mutating step. The exact harness is an implementation choice; what matters is that these invariants are exercised.

### Fork dogfood

Implement and validate against a fork of `rocm-libraries`. Synthetic PRs against the fork exercise the full handler + processor flow on real GitHub APIs, demonstrating the invariants above and edge cases that aren't visible to the algorithm in isolation:

- merge conflict during activation
- CI failure during evaluation
- simultaneous `/merge` on the same PR
- author push between activation and squash

## 7. Rollout

| Phase | What |
|-------|------|
| 0     | Fork dogfood — implement and validate on a fork, covering all invariants and edge cases in [§6](#6-validation). |
| 1     | Commit to rocm-libraries with hipDNN-ecosystem opt-in only. The at-enqueue approval pre-check (see [§5](#5-open-source-contributor-policy)) is enabled from day one. No branch-protection changes. |
| 2     | Add `merge-queue/managed` as **non-blocking** (see [§4.5](#45-pr-lifecycle)). |
| 3     | Promote `merge-queue/managed` to a **required** status check on `develop`. Admins retain bypass as an emergency escape hatch. (See [§8](#8-risks).) |
| 4     | Extend opt-in to other rocm-libraries subprojects on request.                                                   |

## 8. Risks

- **3-minute poll** is slow at low load. Revisit if necessary.
- **Head-of-line stalls.** Cross-queue blocking means a slow upstream PR at the front of multiple queues holds up downstream work.
- **Serialization throughput — wall-clock end-to-end CI for a PR can comfortably exceed several hours.** Three contributing factors:
  - *Per-job timeouts.* Build jobs are bounded at 30 minutes ([`therock-ci-linux.yml`](https://github.com/ROCm/rocm-libraries/blob/develop/.github/workflows/therock-ci-linux.yml), `timeout-minutes: 30`); individual test-component jobs are bounded at 210 minutes / 3.5 hours ([`therock-test-component.yml`](https://github.com/ROCm/rocm-libraries/blob/develop/.github/workflows/therock-test-component.yml), `timeout-minutes: 210`).
  - *Per-cycle setup overhead.* Machine acquisition, cache restoration, and matrix fan-out are fixed per attempt and not amortizable across queue cycles. Runner availability can itself be a bottleneck, adding significant wall-clock time before a job even starts.
  - *n × CI drain time.* Intersecting PRs (e.g. multiple core PRs in succession, an integration-tests PR queued behind providers, or any provider PR queued behind a core or integration-tests PR) merge one at a time, each waiting for a full CI cycle; a queue of *n* intersecting PRs takes roughly *n × CI time* to drain.
- **No CI retries — flakes are terminal.** The processor treats any failed required check as a real failure (`any_required_check_failed → eject`). On a heterogeneous GPU matrix, transient hardware/runner failures eject otherwise-good PRs and force a manual `/merge` re-enqueue, paying another full CI cycle. This compounds with the throughput risk above: every flake-induced eject is another *n × CI* penalty.
- **Surprise auto-eject** when an author pushes new commits mid-queue. Detection happens on the next processor cycle (up to 3 minutes after the push), so feedback is not instant — the eject comment must be explicit about why and how to re-enqueue.
- **Compromised workflow runner.** The tamper-resistance design in [§4.3.1](#431-tamper-resistance) assumes the bot's identity is trustworthy. A compromised `GITHUB_TOKEN` or GitHub App key gives an attacker the same authority as the merge queue itself; the same applies to a sibling workflow in the repo that posts as `github-actions[bot]` — for example, posting a `merge-queue/active` status under that identity would satisfy the bot's activation check. `mq-handler.yml`, `mq-processor.yml`, and any scripts they consume must remain protected by branch protection on the workflow path (already standard for `develop`), and any other workflow that uses `statuses: write` should be reviewed for accidental or deliberate use of merge-queue contexts.
- **Triage-permission misuse.** A user with `triage` repo access can repeatedly remove `mq:*` labels from PRs they don't own, effectively denying service to other contributors' merges. This is a social/permissions problem rather than a queue problem (the same user can already close PRs, request changes, etc.), but each removal is logged via the audit job so abuse is auditable.
- **Silent dequeue from a dropped audit event.** The real-time audit on `pull_request_target: [unlabeled]` is the *only* mechanism for catching `mq:*` label removals — the cycle-time backstop can't see a PR once its labels are gone (the discovery search no longer returns it). If GitHub Actions is degraded, an audit job hits a transient API failure, or the workflow is rate-limited, an `unlabeled` event can be missed and the PR is silently dequeued with no detection or notification. The author has to notice their PR didn't merge and re-`/merge` themselves. No mitigation is engineered around this — the alternative (a durable side-channel for unlabel events) is disproportionate to the failure mode's likelihood and impact.

## Appendix A: Prototype reference

A working prototype implementing a functional resemblance of this design lives at <https://github.com/SamuelReeder/rocm-libraries/tree/develop/.github>. However, this RFC is the source of truth, and the prototype will be reshaped to match these specifications before adoption.
