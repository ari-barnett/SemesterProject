function [newEnv] = stepAgent(x, y, env, gridSize)
    %Calculate Action
    if rand(1) < (1 - log(2))
        newEnv = env;
        newEnv(x,y) = 0;
    else
        newEnv = env;
        position = find_moore(x,y,gridSize,gridSize);
        
        randIndices = randperm(height(position));
        shuffled = position(randIndices, :);
        pos_move = randi(height(position));
        
        newPOS = shuffled(pos_move,:);
        
        newX = newPOS(1);
        newY = newPOS(2);

        if newEnv(newX,newY) == 0
            newEnv(newX,newY) = 1;
        end
    end 
end