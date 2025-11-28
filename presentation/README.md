# Presentation

LaTeX Beamer slides for "Predicting Movie Genre Using Actor Co-Appearance Networks"

## Requirements

- LaTeX distribution (TeX Live, MiKTeX, etc.)
- `beamer` and `metropolis` theme
- XeLaTeX (for custom fonts) or pdfLaTeX

## Building

### Option 1: Make (Default - uses pdflatex)
```bash
make
```

### Option 2: Use xelatex (if you have it configured)
First, generate the format file:
```bash
fmtutil -sys -user -byfmt xelatex
```

Then uncomment `\usepackage{fontspec}` in slides.tex and run:
```bash
make xelatex
```

### Option 3: Direct compilation with pdflatex
```bash
pdflatex slides.tex
pdflatex slides.tex  # Run twice for TOC/references
```

### Option 4: latexmk
```bash
latexmk -pdf slides.tex
```

## Output

Generates `slides.pdf` - a 16:9 aspect ratio presentation with ~15 slides.

## Cleaning

```bash
make clean
```

## Notes

- Images are loaded from `../diagrams/` (the project's diagram folder)
- Uses the Metropolis Beamer theme with custom dark colors
- Run the main `project.py` first to generate the required diagrams

