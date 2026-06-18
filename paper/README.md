# SLHA v2 — arXiv preprint

LaTeX source for the research paper:

> **SLHA v2: Cache-Aware Hybrid Attention with Low-Rank Latents and 1-Bit
> Sign-LSH Residuals for Memory-Bandwidth-Bound LLM Inference**
> *Tarek Zekriti (ZEKRITI TAREK)*

The paper documents the **measured** state of the reference implementation
(`scirust/`): the mechanism, the 128-byte cache-aware tile, CCOS Soft-Paging,
and the full §7 evaluation (throughput, output fidelity, λ calibration, the
quantization-is-not-the-bottleneck ablation). Every number is taken verbatim
from the repository's reproducible (seeded) benchmarks, and the limitations are
stated explicitly (synthetic data, random projections, no real-model perplexity,
no hardware cache counters, no ARM timing).

## Build

Self-contained — only standard LaTeX packages, no external `.bib`, no custom
`.sty`. Two passes resolve cross-references and the bibliography.

```sh
pdflatex slhav2.tex
pdflatex slhav2.tex
```

or, if you have it:

```sh
latexmk -pdf slhav2.tex
```

Builds on Overleaf and on the arXiv submission system as-is. Produces a
13-page PDF.

## arXiv submission

Upload `slhav2.tex` alone (the bibliography is embedded via
`thebibliography`). Suggested primary category: `cs.LG`; cross-list `cs.AR`,
`cs.PF`.
