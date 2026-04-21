# Instructions for Claude Code

## Kanban status sync

When you start work on a GitHub issue (either because the user named it, or
because you picked the next backlog item autonomously), move that issue to
**In Progress** on the `Fantasy Coach — Backlog` project *before* opening the
first PR. When the PR that closes the issue merges to `main`, move the issue
to **Done**. Don't wait for the user to ask — this is how they see what you're
working on.

Commands — IDs are pre-discovered, don't re-look-them-up each turn:

```bash
# Project:       PVT_kwHOAIfoRM4BVIQ8    (lopeztech/projects/10)
# Status field:  PVTSSF_lAHOAIfoRM4BVIQ8zhQmJ_s
# Options:       Todo=f75ad846, In Progress=47fc9ee4, Done=98236657

# Find the project item ID for an issue number:
ITEM_ID=$(gh project item-list 10 --owner lopeztech --format json --limit 200 \
  | jq -r --argjson n "$ISSUE_NUMBER" \
      '.items[] | select(.content.number == $n) | .id')

# Move it. Swap --single-select-option-id for the target column.
gh project item-edit \
  --id "$ITEM_ID" \
  --project-id PVT_kwHOAIfoRM4BVIQ8 \
  --field-id PVTSSF_lAHOAIfoRM4BVIQ8zhQmJ_s \
  --single-select-option-id 47fc9ee4   # In Progress (use 98236657 for Done)
```

If an issue isn't on the board yet (rare — most new issues get auto-added):
`gh project item-add 10 --owner lopeztech --url <issue-url>`.
