# Bilingual paper: seed measurements

The PDF and TeX sources in this directory record the paper update of 26 October
2026 together with its code change. The SA/SC method is on page 18 of
`paper_bilingual.pdf`.

- `M_seed` is the union of retained YOLO seed instance masks.
- `SA = gamma² × foreground pixel count of M_seed` (mm²): overlapping pixels
  contribute once.
- `SC` counts nonempty YOLO seed instances after confidence filtering and
  clipping to the fruit region. Overlap does not merge instances for counting.
- `SCR` uses the same union area in its numerator before display rounding,
  divided by the current endocarp mask area (`M_cavity = M_endocarp`). An empty
  endocarp mask returns zero; the ratio is not clamped to one.

The editable bilingual source is `paper_bilingual.tex`; `source_english.tex`
contains the English version. Images, fonts, and bibliography are included so
the bilingual PDF can be rebuilt from this Git checkout:

```sh
cd docs/paper
tectonic paper_bilingual.tex
```

These files are synchronized with the workspace reading copy in
`../extend/bilingual_paper_20261011/` relative to the repository root. Its
`build_preview.py` and `seed_instance_method_20261026.py` also preserve the union
formula when regenerating the workspace copy. After regeneration, copy the
updated TeX/PDF into this directory to version them with the implementation.

Validation: 26 unit tests passed (seed statistics, runtime export, segmentation
filtering). Replaying the 20 saved mask sets in the workspace comparison report
preserved all 281 YOLO instances and matched the union area in every crop;
18 crops had overlapping pixels that now contribute only once. The updated PDF
was compiled and the method page was visually checked. Existing tables and
dataset-wide results have not been recalculated.

The subsequent SCR denominator update passed all 40 geometric-feature tests,
including seeds and placenta outside endocarp, ratios above one, and empty
endocarp masks. The PDF has been rebuilt with `M_cavity = M_endocarp`; existing
SCR result tables still require recalculation with this denominator.
