% function [anmrrval, nmrrvals] = anmrr(queries, K, NG, rankIndex, version)
%
% computes NMRR (normalized modified retrieval rank) and ANMRR (average
% normalized modified retrieval rank) as defined in [1-5]
%
%   input variables
%
%   queries     cell array with queries and ground truth
%   K           maximum rank for relevant results per query (vector)
%   NG          number of ground truth images (target images) per query (vector)
%   rankIndex   matrix with indicies referring to position in cell array
%               images sorted accortingly the rank of each image
%               example for images 1, 2, and 3:
%
%               1 2 3   query images
%       rank
%        1.     1 2 3   rankIndex for these 3 queries
%        2.     2 3 2
%        3.     3 1 1
%
%               rankIndex has dimensions: length(images) x length(queries)
%
%   version     specifies the version of the ANMRR and NMRR measures as
%               there exist two definitions.
%
%               first   not recommended: NMRR/ANMRR according to [1,3]
%               second  DEFAULT: NMRR/ANMRR according to [4,5]
%
%
%   output variables
%
%   nmrrval     NMRR    vector (number of queries defining the length)
%
%   anmrrval    ANMRR   scalar (1 value)
%
% References:
%
% [1] ISO-MPEG  N2929, October 1999, Melbourne
% [2] Patrick Ndjiki-Nya, Jan Restat, Thomas Meiers, Jens-Rainer Ohm,
%     Anneliese Seyferth, Ruediger Sniehotta, "Subjective Evaluation of the
%     MPEG-7 Retrieval Accuracy Measure (ANMRR)", M6029, May 2000, Geneva
% [3] Wong Ka Man, "Content Based Image Retrieval Using MPEG-7 Dominant
%     Color Descriptor", PhD thesis, September 2004, section 3.4.1.
% [4] B. S. Manjunath and J.-R. Ohm and V.V. Vasudevan and A. Yamada.
%     "Color and texture descriptors", In: IEEE Transactions on
%     Circuits and Systems for Video Technology, vol. 11, no. 6, 2001, pp.
%     703-715
% [5] B. S. Manjunath and Philippe Salembier and Thomas Sikora (Eds.),
%     "Introduction to MPEG-7: Multimedia Content Description
%     Interface", Wiley, 2002
%
function [anmrrval, nmrrvals] = anmrr(queries, K, NG, rankIndex, version)

if (nargin<5)
    version = 'second'; % Default version if version is not specified
end

switch version
    case 'first'
        [anmrrval, nmrrvals] = anmrrFirstDefinition(queries, K, NG, rankIndex);
    case 'second'
        [anmrrval, nmrrvals] = anmrrSecondDefinition(queries, K, NG, rankIndex);
    otherwise
        error('ANMRR: unknown version');
end

% return

function [anmrrval, nmrrval] = anmrrFirstDefinition(queries, K, NG, rankIndex)

Q = numel(queries);
nmrrval = zeros(Q,1);
for q = 1:Q
    % penalty described in M6029, [3]
    Kpenalty = K(q)+1;
    NR(q) = 0;
    for i = 1:numel(queries{q}.targetsIndicies)
        % current rank
        cr = find(rankIndex(:,q) == queries{q}.targetsIndicies(i));
        if (cr <= K(q))
            qRank{q}(i) = cr;
            NR(q) = NR(q) + 1;
        else
            qRank{q}(i) = Kpenalty;
            
            
            %qRank{q}(i) = 1.25 * K(q);
        end
    end
    
    % average rank (AVR)
    avr = sum(qRank{q}) ./ NG(q);
    % modified retrieval rank (MRR)
    mrr = avr - 0.5*(1+NG(q));
    % retrieval rate
    % rr(q) = NR(q) / NG(q);
    % normalized modified retrieval rank
    nmrrval(q) = mrr / (Kpenalty - 0.5*(1+NG(q)));
end
anmrrval = mean(nmrrval);

% queries = 
% NG == label
% k = min{4* 99,2*99}
%
function [anmrrval, nmrrval] = anmrrSecondDefinition(queries, K, NG, rankIndex)

Q = numel(queries);
nmrrval = zeros(Q,1);
for q = 1:Q
    % penalty described in [5] MPEG-7 book (section 12.3) and in [4] 
    Kpenalty = 1.25 * K(q);
    NR(q) = 0;
    for i = 1:numel(queries{q}.targetsIndicies)
        % current rank
        cr = find(rankIndex(:,q) == queries{q}.targetsIndicies(i));
        if (cr <= K(q))
            qRank{q}(i) = cr;
            NR(q) = NR(q) + 1;
        else
            qRank{q}(i) = Kpenalty;
        end
    end
    
    % average rank (AVR)
    avr = sum(qRank{q}) ./ NG(q);
    % modified retrieval rank (MRR)
    mrr = avr - 0.5*(1+NG(q));
    % retrieval rate
    % rr(q) = NR(q) / NG(q);
    % normalized modified retrieval rank
    nmrrval(q) = mrr / (Kpenalty - 0.5*(1+NG(q)));
end
anmrrval = mean(nmrrval);