# Local code graph

Graphify 0.9.100 lives in the repository's isolated, Git-ignored
`.tools/graphify/.venv` environment. This location survives reboots.
The setup commands below need PyPI access. Code-only extraction, clustering,
and queries run locally after installation.

## Install

Run from the repository root:

```sh
python3 -m venv .tools/graphify/.venv
.tools/graphify/.venv/bin/python -m pip --isolated install \
  --no-deps --only-binary=:all: --index-url https://pypi.org/simple \
  'graphifyy==0.9.100' 'networkx>=3.4' 'numpy>=1.21' 'rapidfuzz>=3.0' \
  'pyyaml>=6.0' 'tree-sitter>=0.25,<0.26' 'tree-sitter-python>=0.23,<0.26'
```

The PyPI package is named `graphifyy` (two trailing y's). Its CLI is
`graphify`. This environment includes the dependencies for local Python
extraction, clustering, and queries.

## Rebuild

Run from the repository root:

```sh
export GRAPHIFY_BACKEND=none
export GRAPHIFY_QUERY_LOG_DISABLE=1
GRAPHIFY=.tools/graphify/.venv/bin/graphify
"$GRAPHIFY" extract . --code-only --max-workers 2
"$GRAPHIFY" cluster-only .
```

In this version, extraction writes the graph; `cluster-only` additionally writes
its report and interactive HTML. Backend `none` disables model calls, including
community labeling. Labels may be generic. No trading scripts are executed.

## Query

```sh
GRAPHIFY_BACKEND=none GRAPHIFY_QUERY_LOG_DISABLE=1 \
  .tools/graphify/.venv/bin/graphify \
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
