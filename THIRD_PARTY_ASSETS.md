# Third-party assets and dataset provenance inventory

| Item | Known state | Action |
|---|---|---|
| Cancer-gene/census-derived data | README references curated gene and census-style sources without naming a precise version/license | Record authoritative source, dataset version, access date, citation, and reuse/redistribution permission |
| Gene annotations and derived features | Referenced in project pipeline | Track lineage and any database-specific terms |
| Generated reports | Project reports | Attribute sourced figures/tables; verify any copied content |
| Code and scripts | Original project implementation | MIT covers only rights-held contributions |

**Status:** Dataset identity, edition and redistribution permission not established in this audit. Publicly accessible bioinformatics databases can have attribution or use restrictions.

## Relevant upstream example (2026-10-31)
The COSMIC Cancer Gene Census download page requires registration for noncommercial access and directs commercial users to licensing. **This only applies if this repository actually uses COSMIC data**; the precise dataset used in this repository has not been matched.

- https://cancer.sanger.ac.uk/cosmic/download/cosmic/v104/cancergenecensus

**Action:** trace the exact input data from code and prior downloads to determine whether COSMIC restrictions or a different dataset license apply. No definitive redistribution clearance has been established.
