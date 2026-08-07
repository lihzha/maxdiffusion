# 03 — Every reply states explicitly whether anything is blocked on Yixun

**Directive (Yixun, 2026-08-07, verbatim):**

> Please tell me if something is blocked on me everytime you respond

**Interpretation / how to apply:** every assistant reply carries an explicit **Blocked on you** line — inside the announcement-01 Status block, and never omitted. It answers one question: *is there anything Yixun must do before work can continue?*

- When something IS blocked: name the item, say **exactly what action clears it** (the literal command to run, the decision to make, the credential to refresh), and say **what stops until it clears** versus what continues regardless. If there are several, list them shortest-action-first.
- When nothing is blocked: say so in those words — **"Blocked on you: nothing."** Silence is not an acceptable substitute; the absence of a blocker must be stated as positively as its presence, because "no news" is exactly what this directive exists to eliminate.
- **Distinguish blocked from merely available.** A decision Yixun *may* weigh in on later (an optional report, a deferred arm) is not a blocker; a launch that cannot proceed without a `!` submission, an expired credential, or an unanswered gate decision is. Do not inflate the list — an over-full blocked line hides the real blocker.
- **This is a floor, not a ceiling.** If something becomes blocked mid-turn, say it in that turn; do not wait for a natural stopping point. If a blocker has been open across several replies, keep restating it rather than assuming it was seen.

Rationale: this campaign repeatedly parked on things only Yixun could do — TPU submissions blocked by the permission classifier (issue #10), `gcloud auth login` expiry (issue #6), gate decisions (K2 STOP, J1b/J1c GO), plan approvals. Those are cheap to clear and expensive to leave idle, so they must never be discoverable only by reading the whole reply.
