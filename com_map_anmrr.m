%pca将维
%[pc,score,latent,tsquare] = pca(feats);
%feats=score(:,1:64);

% feats_n=feats./repmat(sqrt(sum(feats.^2,1)),size(feats,1),1);%按列 l2-normalization
NG=99;
Q = 2100;
k=2*NG;
nmrrval = zeros(Q,1);
feats_n=feats./repmat(sqrt(sum(feats.^2,2)),1,size(feats,2));
%feats_n=zscore(feats');
dist=pdist(feats_n,'euclidean');
distance=squareform(mapminmax(dist,0,1));
[value,index]=sort(distance,2);
ap_score=zeros(Q,1);
for q=1:Q
  %{
cr = find(label(q)==label(index(q,2:Q)));
num_class=length(cr) ;%length(query_result);
ap=zeros(num_class,1);
  for j=1:num_class1
    ap(j)=(j)/(cr(j));
  end
  ap_score(q)=sum(ap)/num_class;
  %}
  % penalty described in [5] MPEG-7 book (section 12.3) and in [4] 
    Kpenalty = 1.25 * k;
    
    % current rank
    cr=find(label(q)==label(index(q,2:Q)));
    num_class=length(cr) ;%length(query_result);
    ap=zeros(num_class,1);
    for j=1:num_class
       ap(j)=(j)/(cr(j));
    end
    ap_score(q)=sum(ap)/num_class;
    
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
map=sum(ap_score)/Q;%计算平均Map值
anmrrval = mean(nmrrval);




%搜索查询，前K个p@k的查询准确率

pk=[5,10,50,100,1000];
for p=1:5
    for q=1:Q
        query_result =find(label(q)==label(index(q,2:Q)));
        rr(q)=length(find(query_result<=pk(p)))/pk(p);
    end
    arr(p)=sum(rr)/Q;
end

%Precision-Recall曲线
%precision = 查询到正样本个数/查询的个数
%Recall = 查询正样本个数/总样本数
for i=1:Q
query_result =find(label(i)==label(index(i,2:Q)));
for q=1:Q
    precision(i,q) = length(find(query_result<=q))/q;
    recall(i,q) = length(find(query_result<=q)) / length(query_result);
end
end

precision = sum(precision)/Q;
recall = sum(recall)/Q;
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
title('P-R Curve')


%{
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
%}