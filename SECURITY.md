# Security Policy

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security reports.

- Preferred: open a private report via GitHub
  (**Security → Report a vulnerability** on the
  [medha repository](https://github.com/ArchAI-Labs/medha/security/advisories/new)).
- Alternatively, email the maintainer at **nicola.procopio@acsoftware.it**.

Please include a description, affected version, and a minimal reproduction.
We aim to acknowledge reports within a few business days and to coordinate a
fix and disclosure timeline with you.

## Supported Versions

Security fixes target the **latest released version** on PyPI
(`medha-archai`). Older versions are not maintained.

## Security Model

Medha is a **library** that runs inside the caller's process. It exposes no
network listener and has no authentication surface of its own; its security
posture is defined by the trust boundaries below. There is no use of `pickle`,
`eval`/`exec`, `subprocess`, or `yaml.load` anywhere in the codebase — cache
payloads are serialised as JSON.

### Stored queries are returned verbatim

Medha is a cache: the `generated_query` you store is returned **unchanged** on
a hit. Medha does **not** validate or sanitise the SQL/Cypher/GraphQL you
store, so your application remains responsible for executing returned queries
safely (parameterised execution, least-privilege database credentials, etc.).

Template *parameter values* extracted from a user question **are** sanitised to
`[A-Za-z0-9_\s-]` before being substituted into a `query_template`
(`ParameterExtractor.render_query`), but stored queries are never rewritten.

### Query templates are trusted configuration

`parameter_patterns` in a `QueryTemplate` are regular expressions evaluated
against end-user questions. Load templates only from sources you control: a
hostile or poorly-written pattern can cause catastrophic backtracking (ReDoS).
`max_question_length` (default `8192`) bounds the input size as a partial
mitigation.

### End-user questions

Questions are length-capped (`max_question_length`) and are never interpolated
into backend queries as raw query text. Values reach backends through:

- **parameterised queries** — PostgreSQL/pgvector, VectorChord (`$1`, `$2`, …);
- **typed filter builders** — Weaviate, Elasticsearch, Qdrant, Chroma;
- **centralised literal escaping** — Azure AI Search (OData `$filter`) and
  LanceDB (DataFusion `where`) via `medha.backends._escape.quote_sql_literal`;
- **tag escaping** — Redis Stack (RediSearch) via `_escape_tag`.

### File loading

`warm_from_file()` and `load_templates_from_file()` read from caller-provided
paths and parse **JSON/JSONL only**. When those paths may originate from an
untrusted source, set `allowed_file_dir` to confine reads to a directory, and
tune `max_file_size_mb` (default `100`) to reject oversized files.

### Secrets

All credentials (database passwords, API keys) are typed as Pydantic
`SecretStr`, sourced from environment variables (`MEDHA_*`) or a `.env` file,
and are never logged. Enable transport security where the backend supports it
(e.g. `redis_ssl`, HTTPS Elasticsearch hosts, Qdrant Cloud API keys).

### Backend identifiers

Collection names are sanitised before use in index/table names. For PostgreSQL,
`pg_schema` and `pg_table_prefix` are validated against
`^[a-zA-Z_][a-zA-Z0-9_]{0,62}$`, since identifiers cannot be passed as bind
parameters.
