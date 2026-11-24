TARGET ?= biorxiv
RUN ?= uv run --locked python -B

# The main figures, then the supplementary figures.
FIGURES = analysis_scaling zero_copy_analysis artifact_reuse \
	layout_scaling in_place_access artifact_reuse_measurements reference_mapping_measurements \
	multimodal_graph_agreement

# architecture is drawn in no figure; it prepares the schematic counts that analysis_scaling and zero_copy_analysis read.
FIGURE_PREPARATIONS = architecture $(FIGURES)

.PHONY: all tables figures exports pdfs main supplement source-data clean
all: pdfs exports

tables:
	$(RUN) -m manuscript.tables.render
	mkdir -p publication/tables
	cp .build/tables/table1_approaches.pdf publication/tables/

figures:
	mkdir -p publication/figures
	@set -e; for figure in $(FIGURES); do \
		$(RUN) -m manuscript.figures.$$figure.draw; \
		for suffix in pdf png svg; do cp .build/figures/$$figure/$$figure.$$suffix publication/figures/; done; \
	done

exports:
	mkdir -p publication/tables publication/figure_source_data
	rm -f publication/tables/*.csv
	@set -e; for table in manuscript/tables/*/; do \
		name=$$(basename "$$table"); \
		for input in "$$table"*.csv; do \
			[ -f "$$input" ] || continue; \
			part=$$(basename "$$input" .csv); \
			if [ "$$part" = data ]; then suffix=; else suffix=_$$part; fi; \
			cp "$$input" "publication/tables/$$name$$suffix.csv"; \
		done; \
	done
	@set -e; for figure in $(FIGURES); do \
		mkdir -p publication/figure_source_data/$$figure; \
		rm -f publication/figure_source_data/$$figure/*.json publication/figure_source_data/$$figure/*.npz; \
		for input in manuscript/figures/$$figure/*.json manuscript/figures/$$figure/*.npz; do \
			[ -f "$$input" ] || continue; \
			cp "$$input" publication/figure_source_data/$$figure/; \
		done; \
	done

pdfs: tables figures
	mkdir -p .build publication
	pdflatex -halt-on-error -interaction=batchmode -output-directory=.build "\def\target{$(TARGET)}\input{manuscript/supplementary_information.tex}"
	cd .build && BIBINPUTS=..: BSTINPUTS=..: bibtex supplementary_information
	pdflatex -halt-on-error -interaction=batchmode -output-directory=.build "\def\target{$(TARGET)}\input{manuscript/supplementary_information.tex}"
	pdflatex -halt-on-error -interaction=batchmode -output-directory=.build "\def\target{$(TARGET)}\input{manuscript/main.tex}"
	cd .build && BIBINPUTS=..: BSTINPUTS=..: bibtex main
	pdflatex -halt-on-error -interaction=batchmode -output-directory=.build "\def\target{$(TARGET)}\input{manuscript/main.tex}"
	pdflatex -halt-on-error -interaction=batchmode -output-directory=.build "\def\target{$(TARGET)}\input{manuscript/main.tex}"
	pdflatex -halt-on-error -interaction=batchmode -output-directory=.build "\def\target{$(TARGET)}\input{manuscript/supplementary_information.tex}"
	pdflatex -halt-on-error -interaction=batchmode -output-directory=.build "\def\target{$(TARGET)}\input{manuscript/supplementary_information.tex}"
	cp .build/main.pdf publication/manuscript.pdf
	cp .build/supplementary_information.pdf publication/supplementary_information.pdf

main supplement: pdfs

source-data:
	$(RUN) -m experiments.benchmarks.tool_summary
	$(RUN) -m experiments.benchmarks.analysis_times
	$(RUN) -m experiments.benchmarks.sweep
	$(RUN) -m experiments.benchmarks.remote_comparison
	$(RUN) -m experiments.benchmarks.scaling_workflow_times
	$(RUN) -m experiments.benchmarks.budget_sweep
	$(RUN) -m experiments.benchmarks.store_summary
	$(RUN) -m experiments.benchmarks.count_request_summary
	$(RUN) -m experiments.benchmarks.resource_trace_summary
	$(RUN) -m experiments.integration.multimodal_neighbors_summary
	$(RUN) -m experiments.integration.cite_seq_scaling_summary
	$(RUN) -m experiments.integration.t_cell_reference_summary
	$(RUN) -m experiments.methods.doublet_summary
	$(RUN) -m experiments.catalogue.catalogue_objects_summary
	$(RUN) -m experiments.catalogue.blood_store_screen_summary
	$(RUN) -m experiments.reuse.composite_and_remote_summary
	$(RUN) -m experiments.reuse.progenitor_effects
	$(RUN) -m experiments.interferon_contrasts.lineage_contrasts
	$(RUN) -m experiments.mapping.reference_ladder_summary
	$(RUN) -m experiments.mapping.blood_store_annotation_summary
	$(RUN) -m experiments.mapping.held_out_summary
	$(RUN) -m experiments.mapping.neighbor_count_fork_summary
	$(RUN) -m manuscript.tables.prepare
	@set -e; for figure in $(FIGURE_PREPARATIONS); do $(RUN) -m manuscript.figures.$$figure.prepare; done

clean:
	rm -rf .build
