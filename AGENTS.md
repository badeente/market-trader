# market-trader agent instructions

## Required Graphify-first workflow

For every work session in this repository, consult the Graphify code graph
before analyzing, editing, or reviewing implementation. This is Andreas's
explicit repository workflow preference, not an optional optimization.

1. Read `GRAPHIFY.md` for the local installation and commands. Run commands
   from this repository's root with `GRAPHIFY_BACKEND=none` and
   `GRAPHIFY_QUERY_LOG_DISABLE=1`.
2. Ensure `graphify-out/graph.json` exists and reflects the current code. If
   absent, stale, or freshness is uncertain after a pull/checkout/local change,
   run code-only extraction and `cluster-only` as documented in `GRAPHIFY.md`.
3. Start each task with at least one scoped `graphify query` relevant to the
   task. Use its source-linked nodes and edges to identify the relevant files
   and dependencies. For broad architecture tasks, also consult
   `graphify-out/GRAPH_REPORT.md`.
4. Read the identified source files to verify behavior before making changes.
   Graph relationships guide navigation; they do not replace source review or
   appropriate tests. Use targeted text searches to fill gaps, not as a
   substitute for consulting the graph first.
5. After changing code, refresh extraction and the report before declaring
   the task complete. Re-query the affected area to verify graph coverage.
6. If Graphify is unavailable or fails, attempt a local repair. If it still
   cannot run, explicitly tell Andreas that graph-guided work is unavailable;
   do not silently claim to have followed this workflow.

## Scope and privacy

- Keep graph generation local and code-only with model backends disabled.
- Do not run trading code merely to map the repository.
- Respect `.graphifyignore`; do not publish generated graphs or sensitive data.
- This graph currently excludes notebooks and documentation. If work involves
  these or another unindexed file, consult the graph for related code and then
  inspect the unindexed source directly; state coverage limits when relevant.
