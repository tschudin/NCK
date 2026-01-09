# Makefile

all: main.pdf

main.pdf: main.tex refs.bib fig/noise-gallery-1.pdf fig/noise-gallery-2.pdf
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

fig/noise-gallery-1.pdf: mk_noise_comparison.py
	./mk_noise_comparison.py

clean:
	rm -rf *~ *.aux *.log *.bbl *.blg main.pdf

# eof
