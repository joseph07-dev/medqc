# Security Policy

## Reporting a vulnerability

Open a **GitHub private security advisory** on this repository (Security → Report a
vulnerability), or contact the maintainers through the submission channel. Please do not
file public issues for exploitable vulnerabilities. We aim to acknowledge reports within
48 hours.

## Secrets & credentials

- **No secrets are required to run MedQC** — inference is fully offline.
- The repository must never contain API keys, tokens, or private keys. This is enforced by
  `tests/test_metrics_config.py::test_no_plaintext_secrets_in_tree` (scans the whole tree
  for credential patterns) and by a **gitleaks** step in CI
  (`.github/workflows/ci.yml`), which also checks commit history on push.
- Real secrets for any future integration belong in Streamlit Cloud's *Settings → Secrets*
  or in a git-ignored `.env` (see `.env.example` — placeholders only).

## Threat model for the deployed app

| Risk | Mitigation |
|---|---|
| Malicious/huge uploads | 5 MB cap (`.streamlit/config.toml`), format whitelist (jpg/png/bmp/webp), decode failures caught and shown as errors |
| Upload persistence / data retention | Uploaded images are processed in-memory per request; nothing is written to disk or logged |
| XSS via image metadata or reason codes | Reason codes are drawn only from a fixed server-side enum; captions use bundled filenames only; no user text is rendered as HTML |
| Cross-site request forgery | `enableXsrfProtection = true` (Streamlit default, set explicitly) |
| Telemetry leakage | `browser.gatherUsageStats = false` |
| Dependency compromise | Pinned lower bounds + CI installs on clean runners; onnxruntime/numpy/pillow/streamlit are all mainstream, actively patched packages |

## Deployment hygiene

- Fully local inference — no cloud deployment, no outbound requests at runtime.
- Repository secret scanning / push protection should be enabled in GitHub settings.

## Supported versions

Only the latest commit on `main` is supported.
