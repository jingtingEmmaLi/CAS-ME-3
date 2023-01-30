clear; clc;
filepath = './**inputpath**/';  % assuming there are several ME samples to deal with
list = dir([filepath,'*.mat']);
for ii = 1:length(list)
    savepath = ['./**outputpath**/', list(ii).name];
    if ~exist(savepath,'file')
        disp(list(ii).name)
        load([filepath,list(ii).name]);
        onsetG = gpuArray(onset);
        apexG = gpuArray(apex);
        clear onset apex
        [ux,uy,uz]=LKPR3D(onsetG,apexG,4,2,0,1);
        % Enhance the quiver plot visually by downsizing vectors  
        %   -f : downsizing factor
    %     f=5;
    %     x=ux(1:f:size(ux,1),1:f:size(ux,2),1:f:size(ux,3)); 
    %     y=uy(1:f:size(ux,1),1:f:size(ux,2),1:f:size(ux,3)); 
    %     z=uz(1:f:size(ux,1),1:f:size(ux,2),1:f:size(ux,3)); 
    % 
    %     [X,Y,Z]=meshgrid(1:size(x,2),1:size(x,1),1:size(x,3));
    %     quiver3(X,Y,Z,x,y,z); axis([1 size(x,2) 1 size(x,1) 1 size(x,3)]);

    %%

        save(savepath,'ux','uy','uz');
        disp('saved')

        clear onsetG apexG ux uy uz
    end
end
