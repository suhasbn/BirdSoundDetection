% db3_41feats.mat
% db4_54feats.mat
% db5_45feats.mat
% mel_64feats.mat
% mfbe_51feats.mat
% mfcc_55feats.mat
% sym4_54feats.mat
% sym5_54feats.mat
% sym6_54feats.mat
load('Normal/sym5_54feats.mat');
number_of_feats = 54;
feat_name = 'sym5';
%data = paper5;
noise = {'0.2','0.4','0.6','0.8'};
for i = 1:length(noise)
    data_temp = data./max(data, [], 'all');
    split_hasBird = data_temp(1:1935,:);
    split_noBird = data_temp(1936:end,:);
    split_hasBird_noise = split_hasBird(1:1935*str2num(char(noise(i))),:) + randn(1935*str2num(char(noise(i))),number_of_feats);
    split_hasBird_normal = split_hasBird(1935*str2num(char(noise(i)))+1:1935,:);
    yes = [split_hasBird_noise;split_hasBird_normal];

    split_noBird_noise = split_noBird(1:1935*str2num(char(noise(i))),:) + randn(1935*str2num(char(noise(i))),number_of_feats);
    split_noBird_normal = split_noBird(1935*str2num(char(noise(i)))+1:1935,:);
    no = [split_noBird_noise;split_noBird_normal];

    noisySignal = [yes;no];
    name = strcat('updated_',char(noise(i)),'/',char(noise(i)),'_',feat_name,'_noisySignal.mat');
    save(name,'noisySignal');
    %plot(noisySignal(157,:));hold on;plot(data(157,:));hold off;legend('noisy','original');
end