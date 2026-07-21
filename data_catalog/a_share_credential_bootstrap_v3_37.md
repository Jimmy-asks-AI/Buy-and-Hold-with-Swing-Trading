# A-share Credential Bootstrap V3.37

Updated: `2026-05-27T21:53:16`

This catalog entry records credential hygiene for PIT data acquisition.

## Provider Readiness

- Tushare ready: `False`
- JoinQuant ready: `False`

## Secret Policy

- Real credentials belong only in environment variables or local `configs/data_credentials.json`.
- `configs/data_credentials.json` is gitignored.
- Output artifacts contain boolean readiness only, never secret values.

## Policy Checks

| Check | Status |
|---|---:|
| `data_credentials_json_gitignored` | `pass` |
| `local_json_pattern_gitignored` | `pass` |
| `env_local_gitignored` | `pass` |
| `local_template_exists` | `pass` |
| `local_template_placeholders_only` | `pass` |
| `execute_example_exists` | `pass` |
| `real_credential_file_exists` | `blocked` |
