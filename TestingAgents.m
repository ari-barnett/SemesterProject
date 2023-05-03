GAN_DATA = [];
img_thing = {};
for simulations = 1:50
    q = 1;
    close all; clc;
    % Parameters
    disp("Simulation #" + simulations)
    nRabbits = randi(10);
    nSteps = 100;
    gridSize = 200;
    data = zeros(nSteps,1);
    custom_colormap = [0, 0, 0; 1, 0, 0; 0, 0, 1];
    
    % Environment
    %load("tons.mat")
    env = zeros(gridSize, gridSize);
    
    % Place agents randomly
    for i = 1:nRabbits
        x = randi(gridSize);
        y = randi(gridSize);
        env(x, y) = 1; 
    end
    
    figure()
    colormap(custom_colormap);
    for step = 1:nSteps
        %tic
        % Shuffle the order of cells to update
        [rows, cols] = find(env == 1);
        idx = randperm(length(rows));
        rows = rows(idx);
        cols = cols(idx);
        
        % Move and interact
        for i = 1:length(rows)
            x = rows(i);
            y = cols(i);
            newEnv = stepAgent(x, y, env, gridSize);
            env = newEnv;
        end
        
        % Display the environment
        %subplot(2,1,1)
        if step == 1 || step == 100 || step == 200 || step == 300
            img_thing{q} = env;
            q = q+1;
           % treatment_effective = randi(10000,[5000 2]);
            %disp("TREATMENT APPLIED")
            %env(treatment_effective) = 0;

        end
            imagesc(env);
            %env_show = repmat(env,[1 1 3]);
            %imshow(env_show,'InitialMagnification',1000)
            %im2bw(env);
            %imshow(env,'Colormap',winter(256))
            pause(0.05);
    
        data(step,1) = sum(reshape(newEnv, 1, []));
        data(step,2) = sum(reshape(newEnv(1:gridSize/2,1:gridSize/2), 1, []));
        data(step,3) = sum(reshape(newEnv((gridSize/2) + 1:gridSize,1:gridSize/2), 1, []));
        data(step,4) = sum(reshape(newEnv(1:gridSize/2,(gridSize/2) + 1:gridSize), 1, []));
        data(step,5) = sum(reshape(newEnv((gridSize/2) + 1:gridSize,(gridSize/2) + 1:gridSize), 1, []));
    
        %toc
        %subplot(2,1,2)
        %scatter(step,sum(sum(env)),'black','filled')
        %hold on
    
    end
    figure()
    plot((1:nSteps)',data)
    GAN_DATA = [GAN_DATA; data(:,1)];
end
disp("SIMULATION COMPLETE")
save("FILE.mat", "GAN_DATA")