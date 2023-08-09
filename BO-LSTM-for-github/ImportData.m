function data = ImportData(num_start, num, N_oneCPT)
qc_t = zeros(N_oneCPT,1);
data = [];
for i = num_start:1:num
    initial_data = readmatrix('original data.xlsx','Sheet', strcat('Sheet',num2str(i)));  %% loading original dataset
    
    Dr = initial_data(1,1);         % relative density
    sigma_v = initial_data(1,2);    % effective vertical stress
    K0 = initial_data(1,3);         % lateral earth pressure coefficient
    saturation = initial_data(1,4); % -1 represents dry sand; 0 represents saturated sand; 
    pene_vel = initial_data(1,5);   % vertical penetrtion rate, m/s;
    BC = initial_data(1,6);         % boundary conditions, BC1-BC5;
    
    depth_qc = [zeros(1,2);initial_data(:,7:8)]; % depth-qc response, unit: cm-MPa. Add 0,0 point to avoid singularity
    
    depth = depth_qc(end,1);                    % total penetation depth 
    depth_qc = [depth_qc(:,1)/depth,depth_qc(:,2)];

    depth_i = depth_qc(1,1):1/N_oneCPT:1; % depth increment
    
    ExpData = interp1(depth_qc(:,1), depth_qc(:,2), depth_i');  % use linear interpolation to process one qc curve 
    ExpData = [depth_i', ExpData];
    ExpData(1,:) = [];                           % remove the added 0,0 point
    
    smooth_qc = smoothdata(ExpData(:,end));      % data smooth
    ExpData = [ExpData(:,1), smooth_qc];
    
    output_data = [repelem(Dr,N_oneCPT)',repelem(sigma_v,N_oneCPT)',repelem(K0,N_oneCPT)', ...
        repelem(saturation,N_oneCPT)',repelem(BC,N_oneCPT)',ExpData];  % put input and output data in one matrix
    data = [data;output_data];                   % all training and testing dataset
end

end
