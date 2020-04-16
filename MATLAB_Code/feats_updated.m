clc;
clear;
close;


path=('/home/suhas/Desktop/Sem2/audio-classification/data_backup/');
cd(path);
files = dir(path);
wavelet = {'db1','db2','db3','db4','db5','sym1','sym2','sym3','sym4','sym5'};
count = 0;
data = [];
for i =1:length(wavelet)
    count = 0;
    for file = 3:length(files)
       name = files(file).name;
       temp = audioread(name);
       [filter_db4,l] = wavedec(temp,6,char(wavelet(i)));
       [cD6] = detcoef(filter_db4,l,[6]);
       data = [data;cD6];
       count = count+1;
       disp(count);
    end
    path = strcat('/home/suhas/Desktop/Sem2/audio-classification/NEW/',string(char(wavelet(i))),'.mat')
    save(path,'data')
end
    %save('/home/suhas/Desktop/Sem2/audio-classification/data_backup/db4_noBird.mat','data');
%179 for 8 and 17 for 12
%data = reshape(data,[114,1935]);