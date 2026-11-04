function phic = phiReplace(phic_str, interval_LR, T_l, T_r)
% phiReplace function returns the phi with a custom T
%
% Inputs:
%   phic_str: the string of the dynamic specification
%   interval_LR: the interval specified in the specification (for intersection)
%   T_l: the first instance when the robustness value turns negative
%   T_r: the first instance when the negative robustness dispears or the instance when the simulation ends
% Outputs:
%   phic: the specification with T_l and T_r

% T_l should be less than T_r
if T_l > T_r
    error('T_l should be less than T_r.');
end

phic_replace = '';
% find the correct interval_LR{1,i} that T_l and T_r belong to
for i = 1:numel(interval_LR)
    % if the violation episode is later than interval_LR{1,i}, replace T_begin and T_end with interval_LR{1,i}
    if T_l > interval_LR{1,i}(1,2)
        sub_phic_replace_l = strrep(phic_str{1,i}, 'T_begin', num2str(interval_LR{1,i}(1,1)));
        sub_phic_replace_r = strrep(sub_phic_replace_l, 'T_end', num2str(interval_LR{1,i}(1,2)));
        phic_replace = strcat(phic_replace, sub_phic_replace_r);
        phic_replace = strcat(phic_replace, ' and ');
    % if the violation episode falls between current interval_LR  replace T_begin with interval_LR{1,i}(1,1), replace T_end with T_r
    elseif T_l >= interval_LR{1,i}(1,1) && T_r <= interval_LR{1,i}(1,2)
        sub_phic_replace_l = strrep(phic_str{1,i}, 'T_begin', num2str(interval_LR{1,i}(1,1)));
        % Write T_r exactly: num2str keeps 4 significant digits, so a sample
        % time such as 146*0.1 = 14.600000000000001 became '14.6', just below
        % the sample, and the last sample of the window went unchecked.
        sub_phic_replace_r = strrep(sub_phic_replace_l, 'T_end', sprintf('%.17g', T_r));
        phic_replace = strcat(phic_replace, sub_phic_replace_r);
        break;
    elseif T_l < interval_LR{1,i}(1,1) || T_r > interval_LR{1,i}(1,2)
        error('T_l or T_r exceeds bound!');
    end
end

phic = STL_Formula('phic', phic_replace);
end