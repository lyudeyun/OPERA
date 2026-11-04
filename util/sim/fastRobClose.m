function fastRobClose()
% FASTROBCLOSE  Suspend this worker's simulation session: the model comes out
% of fast restart, but the session stays cached.
%
% A model left in fast restart makes MATLAB prompt before closing it, which
% fails a batch job at exit, so the search calls this when a search finishes.
% That happens once per fitness mode per budget -- dozens of times in one
% experiment -- so this must not discard the session; rebuilding the model and
% the Breach interface that often would cost more than the whole speedup.
%
% Called on the client with a pool running, it suspends the workers too.

if isempty(getCurrentTask()) && ~isempty(gcp('nocreate'))
    wait(parfevalOnAll(@fastRobClose, 0));
end
Simulink.sdi.clear   % fastRob only clears in batches, so drop the last one
fastRobOpen();       % no arguments: suspend, keeping the session cached
end
