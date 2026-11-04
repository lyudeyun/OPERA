# Runs every job script in test/scripts, written by the generators in test/; the output of each job
# is logged to output/<job>.log. 'make clean' removes the files Simulink leaves in the root.

JOBS := $(notdir $(wildcard test/scripts/*))

.PHONY: all clean $(JOBS)

all: $(JOBS)

$(JOBS):
	./run $@ test/scripts/$@

clean:
	rm -f *.dat *.slxc *mex*
