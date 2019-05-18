

feats_n=feats./repmat(sqrt(sum(feats.^2,2)),1,size(feats,2));
%feats_n=zscore(feats');
dist=pdist(feats_n,'euclidean');
distance=squareform(mapminmax(dist,0,1));
[value,index]=sort(distance,2);

NG=99;
Q = 2100;
k=2*NG;
nmrrval = zeros(Q,1);


%搜索查询，前K个p@k的查询准确率

pk=5;
for i=1:Q
    query_result =find(label(i)==label(index(i,2:2100)));
    rr(i)=length(find(query_result<=pk))/pk;
end
arr=sum(rr)/Q;





for q = 1:Q
    % penalty described in [5] MPEG-7 book (section 12.3) and in [4] 
    Kpenalty = 1.25 * k;
    
    % current rank
    cr=find(label(q)==label(index(q,2:Q)));
    a(q)=length(find(cr<=k));
   
    qRank(q) = sum(cr(find(cr<=k)))+1.25 * (k+1)*(NG-length(find(cr<=k)));
    
    %qRank(q) =  Kpenalty;
    % average rank (AVR)
    avr(q) = qRank(q)/ NG;
    % modified retrieval rank (MRR)
    mrr = avr(q) - 0.5*(1+NG);
    % retrieval rate
    % rr(q) = NR(q) / NG(q);
    % normalized modified retrieval rank
    nmrrval(q) = mrr / (Kpenalty - 0.5*(1+NG));
end
anmrrval = mean(nmrrval);