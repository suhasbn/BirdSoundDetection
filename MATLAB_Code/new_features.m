clc;
clear;
close;


path=('/home/suhas/Desktop/Sem2/audio-classification/data_backup/');
cd(path);
files = dir(path);
wavelet = {'sym4','sym5','sym6'};
count = 0;
data = [];
for i =3:length(wavelet)
    count = 0;
    for file = 3:length(files)
       name = files(file).name;
       temp = audioread(name);
       %temp = temp/max(temp);
       [filter_db4,l] = wavedec(temp,6,char(wavelet(i)));
       [cD6] = detcoef(filter_db4,l,[6]);
       %avge = mean(cD6);
       %med = median(cD6);
       %stdev = std(cD6);
       %energy1 = wenergy(cD6,1);energy2 = wenergy(cD6,2);energy3 = wenergy(cD6,3);energy4 = wenergy(cD6,4);energy5 = wenergy(cD6,5);energy6 = wenergy(cD6,6);

       %energy = [avge;med;stdev;energy1;energy2;energy3;energy4;energy5;energy6];

       
       data = [data;cD6];
       count = count+1;
       disp(count);
    end
    data = reshape(data,[3870,length(cD6)])
    path = strcat('/home/suhas/Desktop/Sem2/audio-classification/NEW/',string(char(wavelet(i))),'.mat')
    save(path,'data')
end
    %save('/home/suhas/Desktop/Sem2/audio-classification/data_backup/db4_noBird.mat','data');
%179 for 8 and 17 for 12
%data = reshape(data,[114,1935]);
