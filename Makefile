# Get all script paths under test/scripts
SCRIPTS=$(wildcard test/scripts/*)
# Extract filenames as targets (e.g., test/scripts/foo -> foo)
TARGETS=$(notdir $(SCRIPTS))

.PHONY: all clean output $(TARGETS)

# Default target: run all scripts
all: $(TARGETS)

# Ensure output directory exists
output:
	mkdir -p $@

# Rule: Run the script with the specified name
# Usage: make <script_name> or make all
# $* is the target name (script name), $< is the prerequisite (script path)
$(TARGETS): %: test/scripts/% output
	./run $* $<

clean:
	rm -f *.dat *.slxc *mex*
	rm -rf output
