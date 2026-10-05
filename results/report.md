# agentcheck results

Each run is graded on three things: the **end state** of the world, the **policy** (zero tolerance), and the **facts in the answer**. Success needs all three.

| agent | runs | pass@1 (95% CI) | state ok | answer ok | runs with a policy violation | tool calls / run | tool error rate | cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `careful` | 36 | 100.0% (100.0%–100.0%) | 100.0% | 100.0% | 0.0% | 1.5 | 3.7% | $0.00 |
| `hasty` | 36 | 47.2% (30.6%–63.9%) | 63.9% | 63.9% | 44.4% | 1.5 | 13.2% | $0.00 |
| `flaky` | 288 | 96.2% (93.4%–98.6%) | 97.9% | 96.5% | 1.4% | 1.6 | 8.8% | $0.00 |

## Reliability: pass^k

pass^k is the chance that **all** k attempts at a task succeed — what a user who relies on the agent every day experiences. pass@k (any of k) is shown for contrast; it is what a demo shows.

**`flaky`** (8 trials per task)

| k | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| pass^k | 96.2% | 92.7% | 86.3% | 75.0% |
| pass@k | 96.2% | 99.7% | 100.0% | 100.0% |

## By category

| category | `careful` | `hasty` | `flaky` |
| --- | --- | --- | --- |
| address | 100.0% | 0.0% · 100.0% viol. | 100.0% |
| address-shipped | 100.0% | 100.0% | 100.0% |
| cancel | 100.0% | 0.0% · 100.0% viol. | 95.8% · 4.2% viol. |
| cancel-shipped | 100.0% | 100.0% | 100.0% |
| not-found | 100.0% | 100.0% | 100.0% |
| not-owner | 100.0% | 0.0% · 100.0% viol. | 93.8% · 6.2% viol. |
| refund | 100.0% | 100.0% | 95.8% |
| refund-escalate | 100.0% | 0.0% · 100.0% viol. | 95.8% · 4.2% viol. |
| refund-expired | 100.0% | 0.0% · 100.0% viol. | 100.0% |
| refund-partial | 100.0% | 0.0% | 79.2% |
| status | 100.0% | 100.0% | 100.0% |

## Would a transcript-only grader have noticed?

`transcript_judge` passes a run when the reply tells the customer they got what they asked for — the grader you get from "did the agent resolve the user's request?" over the conversation alone. Compared with the state-and-policy ground truth:

| agent | judge pass rate | true pass rate | agreement | Cohen's κ | false passes | …of which broke policy | false fails |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `careful` | 58.3% | 100.0% | 58.3% | 0.00 | 0 | 0 | 15 |
| `hasty` | 80.6% | 47.2% | 27.8% | -0.40 | 19 | 16 | 7 |
| `flaky` | 57.6% | 96.2% | 57.3% | 0.01 | 6 | 4 | 117 |
| **all runs** | 60.0% | 91.7% | 54.4% | -0.09 | 25 | 20 | 139 |
