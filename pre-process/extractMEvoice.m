clc;
close all;
clear all
%%
labels = importdata('mockCrime_label.mat');
Fs = 48000;
voice_path = './original/voice/'
files = dir(voice_path);
    % 文件数量
len = length(files);
fps = 30;
sampleRate = Fs/fps;
%%
index_label = 1;
for ii = 1:len   
    if (strcmp(files(ii).name, '.') == 1) ...
                || (strcmp(files(ii).name, '..') == 1)
            continue;
    end
    newStr = split(files(ii).name,'.') ;
    index = str2double(newStr{1,1});
    if find(labels(:,1),index)
        fileId = fopen([voice_path,files(ii).name],'r');
        x = fread(fileId,inf,'int16');
        wavelength = size(x,1);
        ch1 = x(1:wavelength/2);
        ch2 = x(wavelength/2+1:end);
        
        while index_label<= size(labels,1) && index == labels(index_label,1)
            
            if int32(labels(index_label,3)*sampleRate) < wavelength/2
                start = int32(labels(index_label,3)*sampleRate);            
                if int32(labels(index_label,4)*sampleRate) <= wavelength/2
                    final = int32(labels(index_label,4)*sampleRate);
                else
                    final = wavelength/2;
                end
                me_voice = zeros(2,final-start+1);     
                me_voice(1,:) = ch1(start:final);            
                me_voice(2,:) = ch2(start:final);
                save(['./preprocess/voice/',num2str(index),'_',num2str(labels(index_label,2)),'.mat'],'me_voice');                          
            end
            index_label = index_label + 1; 
            clear start final
        end
        
    end  
end 

