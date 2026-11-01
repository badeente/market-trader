# Local code graph

Graphify 0.9.100 is installed in an isolated environment outside this repository:
`/home/harder/.openclaw/workspace-dividend/.venvs/graphify`.

## Rebuild

Run from the repository root:

```sh
export GRAPHIFY_BACKEND=none
export GRAPHIFY_QUERY_LOG_DISABLE=1
GRAPHIFY=/home/harder/.openclaw/workspace-dividend/.venvs/graphify/bin/graphify
"$GRAPHIFY" extract . --code-only --max-workers 2
"$GRAPHIFY" cluster-only .
```

In this version, extraction writes the graph; `cluster-only` additionally writes
its report and interactive HTML. Backend `none` disables model calls, including
community labeling. Labels may be generic. No trading scripts are executed.

## Query

```sh
GRAPHIFY_BACKEND=none GRAPHIFY_QUERY_LOG_DISABLE=1 \
  /home/harder/.openclaw/workspace-dividend/.venvs/graphify/bin/graphify \
  query 'How do strategies connect to the backtester?'
```

Queries return source-linked subgraphs, not a complete natural-language answer.
Check source files when interpreting inferred relationships.

## Outputs and scope

- `graphify-out/graph.json`: graph data.
- `graphify-out/GRAPH_REPORT.md`: structural report.
- `graphify-out/graph.html`: interactive browser view; generated, not browser-verified.

Output is ignored locally through `.git/info/exclude`; it is not published.
`.graphifyignore` excludes caches, common data/model artifacts, and environment
files while retaining Python utility scripts in data directories.

The initial scan covered 34 Python files and produced 295 nodes, 533 edges, and
14 communities. The two notebooks and documentation were not indexed. Five
files yielded no symbols; a successful scan does not imply complete runtime
call coverage. Code-only extraction uses local parsing and no model API key.

Upstream: https://github.com/Graphify-Labs/graphify
