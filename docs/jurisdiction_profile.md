# Jurisdiction Profile

Adding a new jurisdiction = writing one YAML file under
`configs/jurisdictions/<iso_code>.yaml`. No code change is required for
collection / extraction / classification — the pipeline reads the profile
to discover sources and language settings.

## Schema

```yaml
jurisdiction: "Singapore"        # human-readable name
iso_code: "SG"                   # ISO 3166-1 alpha-2 (or alpha-3)
primary_language: "en"           # BCP 47
additional_languages: []         # e.g. ["zh", "ms", "ta"]
legal_system: "common"           # one of: civil | common | hybrid

# OCR languages to load for scanned documents (Tesseract codes)
ocr_languages: ["eng"]

# Keyword expansions for Pillar 6/7 retrieval, by language.
# These augment the global lexicon in configs/rdtii_indicators.yaml.
keywords_by_indicator:
  "6.1":
    en: ["cross-border transfer", "transfer abroad", "overseas transfer"]
  "7.1":
    en: ["personal data", "data protection officer"]

# Source portals — listed in priority order. Each portal MUST declare source_type.
portals:
  - name: "Singapore Statutes Online"
    url: "https://sso.agc.gov.sg/"
    source_type: "primary"        # primary | secondary
    fetch_method: "http"          # http | sitemap | playwright | api
    search_query: "personal data protection"
    notes: "Official consolidated text of statutes."
```

## Source-type rules

- `primary` — binding laws, regulations, official gazettes, ratified treaties.
  Outputs from these portals are **citable evidence**.
- `secondary` — ministry guidelines, FAQs, commentary, third-party datasets.
  Outputs are **discovery context only**. They never appear as a binding
  citation unless they themselves reproduce and link to a primary instrument.

## Adding a portal

1. Decide its `source_type` honestly. When in doubt, ask: would a court accept
   a quote from this URL as binding law? If no → `secondary`.
2. Choose the least brittle `fetch_method` available (`api` > `sitemap` >
   `http` > `playwright`).
3. Test with `lexora collect --jurisdiction <iso_code> --dry-run`.

## Currently shipped profiles

- `sg.yaml` — Singapore
- `jp.yaml` — Japan
- `th.yaml` — Thailand

Use `_template.yaml` as a starting point.
