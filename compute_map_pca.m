function [pcaA V] = fastPCA( A, k )
% 快速PCA
% 输入：A --- 样本矩阵，每行为一个样本
%      k --- 降维至 k 维
% 输出：pcaA --- 降维后的 k 维样本特征向量组成的矩阵，每行一个样本，列数 k 为降维后的样本特征维数
%      V --- 主成分向量
[r c] = size(A);
% 样本均值
meanVec = mean(A);
% 计算协方差矩阵的转置 covMatT
Z = (A-repmat(meanVec, r, 1));
covMatT = Z * Z';
% 计算 covMatT 的前 k 个本征值和本征向量
[V D] = eigs(covMatT, k);
% 得到协方差矩阵 (covMatT)' 的本征向量
V = Z' * V;
% 本征向量归一化为单位本征向量
for i=1:k
    V(:,i)=V(:,i)/norm(V(:,i));
end
% 线性变换（投影）降维至 k 维
pcaA = Z * V;
% 保存变换矩阵 V 和变换原点 meanVec
end






%pca降维
%feature=feature';
[pc,score,latent,tsquare] = pca(feature);%我们这里需要他的pc和latent值做分析

%cumsum(latent)./sum(latent);%latent用来计算降维后取多少维度能够达到自己需要的精度, 
%tran=pc(:,1:128);%我们需要取pc中的1:50列来做最后的变换矩阵:

%feature= bsxfun(@minus,feature,mean(feature,1));
%feats= feature*tran;
feats=score(:,1:128); %和以上步骤等价

%u=平均值 b=标准差   （1+（x-mean(x,2)）/(3*std(x,0,2))）
feats_n=zeros(2100,4096);
aa = feats-mean(feats,2);
for i=1:2100
for j=1:4096
  
  feats_n(i,j) = 1/2+(aa(i,j))/(6*std(feats(i,:)));
end
end
%z-score归一化 检索map值最低
feats_n=mapminmax(zscore(feats,0,2),0,1);


% feats_n=feats./repmat(sqrt(sum(feats.^2,1)),size(feats,1),1);%按列 l2-normalization
nums=2100;
feats_n=feats./repmat(sqrt(sum(feats.^2,2)),1,size(feats,2));
%feats_n=zscore(feats');
dist=pdist(feats_n,'euclidean');
distance=squareform(mapminmax(dist,0,1));
[value,index]=sort(distance,2);
ap_score=zeros(nums,1);
for i=1:nums
query_result =find(label(i)==label(index(i,2:nums)));
num_class=length(query_result);
ap=zeros(num_class,1);
  for j=1:num_class
    ap(j)=(j)/(query_result(j));
  end
  ap_score(i)=sum(ap)/num_class;
end
map=sum(ap_score)/nums;%计算平均Map值


%p@k(k=5,10,15,....)计算map值

% feats_n=feats./repmat(sqrt(sum(feats.^2,1)),size(feats,1),1);%按列 l2-normalization
nums=2100;
feats_n=feats./repmat(sqrt(sum(feats.^2,2)),1,size(feats,2));
%feats_n=zscore(feats');
dist=pdist(feats_n,'euclidean');
distance=squareform(mapminmax(dist,0,1));
[value,index]=sort(distance,2);
ap_score=zeros(nums,1);
for i=1:nums
query_result =find(label(i)==label(index(i,2:nums)));
num_class=50 ;%length(query_result);
ap=zeros(num_class,1);
  for j=1:num_class
    ap(j)=(j)/(query_result(j));
  end
  ap_score(i)=sum(ap)/num_class;
end
map=sum(ap_score)/nums;%计算平均Map值



dist=pdist(feats,'euclidean');
distance=squareform(dist);
[value,index]=sort(distance,2);
ap_score=zeros(6080,1);
for i=1:6080
query_result =find(label(i)==label(index(i,2:6080)));
ap=zeros(159,1);
  for j=1:159
    ap(j)=(j)/(query_result(j));
  end
  ap_score(i)=sum(ap)/159;
end
map=sum(ap_score)/6080;%计算平均Map值




%每一幅图像轮流做查询图像并计算AP
%1、对提取的特征进行以下处理
%resnet
resnet_n=resnet./repmat(sqrt(sum(resnet.^2,2)),1,size(resnet,2));%按行标准化
resnet_d=pdist(resnet_n,'euclidean');%计算特征两两之间的欧式距离
resnet_d=squareform(mapminmax(resnet_d,0,1));%[0,1]归一化
%caffenet
caffenet_n=caffenet./repmat(sqrt(sum(caffenet.^2,2)),1,size(caffenet,2));%按行准化
caffenet_d=pdist(caffenet_n,'euclidean');%计算特征两两之间的欧式距离
caffenet_d=squareform(mapminmax(caffenet_d,0,1));%[0,1]归一化
%google
google_n=google./repmat(sqrt(sum(google.^2,2)),1,size(google,2));%按行标准化
google_d=pdist(google_n,'euclidean');%算特征两两之间的欧式距离
google_d=squareform(mapminmax(google_d,0,1));%[0,1]归一化
%vggs
vggs_n=vggs./repmat(sqrt(sum(vggs.^2,2)),1,size(vggs,2));%按行标准化
vggs_d=pdist(vggs_n,'euclidean');%计算特征两两之间的欧式距离
vggs_d=squareform(mapminmax(vggs_d,0,1));%[0,1]归一化
%alexnet
alexnet_n=alexnet./repmat(sqrt(sum(alexnet.^2,2)),1,size(alexnet,2));%按行标准化
alexnet_d=pdist(alexnet_n,'euclidean');%计算特征两两之间的欧式距离
alexnet_d=squareform(mapminmax(alexnet_d,0,1));%[0,1]归一化
%多专家特征融合
g_c=(google_d+resnet_d+caffenet_d)/3;
gc_d=pdist(g_c,'euclidean');
gc_d=squareform(mapminmax(gc_d,0,1));


gc_d=(google_d+resnet_d+caffenet_d);
%多个和单个专家计算MAP值
[value,index]=sort(google_d,2);
map=zeros(6080,1);
for i=1:6080
query_result =find(label(i)==label(index(i,2:6080)));
ap=zeros(159,1);
  for j=1:159
    ap(j)=(j)/(query_result(j));
  end
  map(i)=sum(ap)/159;
end
map=sum(map)/6080;%计算平均Map值




map_d=zeros(11,1);
for z=0:10
%多个专家特征以不同的权重相加融合
gc_d=(resnet_d*0.1*z+google_d*(1-(0.1*z)));
%gc_d=squareform(gc_d);
%多个和单个专家计算MAP值
[value,index]=sort(gc_d,2);
map=zeros(6080,1);
for i=1:6080
query_result =find(label(i)==label(index(i,2:6080)));
ap=zeros(159,1);
  for j=1:159
    ap(j)=(j)/(query_result(j));
  end
  map(i)=sum(ap)/159;
end
map_d(z+1)=sum(map)/6080;%计算平均Map值
end