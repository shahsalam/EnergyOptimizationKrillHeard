clc
clear
format short G
%% Problem setting 
%% Group 1
ub = [25,94,107,371];       % upper bound
lb= [22,45,60,310];         % lower bound  
%% Group 2
%ub = [25,94,162,371];       % upper bound
%lb= [23,49,60,324];         % lower bound  
%% Group 3
%ub = [26,94,162,371];       % upper bound
%lb= [25,87,60,322];         % lower bound  
%% Minimum 
%ub=[25,94,80,371];
%lb=[21,41,60,310];
%% Maximum 
%ub=[26,94,164,371];
%lb=[25,87,60,324];
%% Average 
%lb=[25,94,129,371];
%ub=[23,58,60,318];
Food = 'TrainingNewDataUserGroup1 - Copy.xlsx';
% filename = 'Data_knn.xlsx';
data2 = xlsread(Food);
global eT eH eL eA

prob = @cIndex1 ;       %fitness function
prob1 = @calculateFitness;
%% Group 1
%bounds = [[22; 25], [45; 94], [60; 107],[310; 371]]; % bounds in array form  %60

%% Group 2
bounds = [[23; 25], [49; 94], [60; 162],[324; 371]]; % bounds in array form

%% Group 3
%bounds = [[25; 26], [87; 94], [60; 162],[322; 371]]; % bounds in array form

%% Mainimum
%bounds = [[21; 25], [41; 94], [60; 80],[310; 371]]; % bounds in array form

%% Maximum
%bounds = [[25; 26], [87; 94], [60; 164],[324; 371]]; % bounds in array form
%% Average
%bounds = [[23; 25], [58; 94], [60; 129],[318; 371]]; % bounds in array form

%% Algorithm parameter
                                       % population size
max_iter = 50;
Nmax = 0.01;                                   %maximum induced speed
Nold = 0;                                      % last induced motion induced
vf = 0.02;                                      % foraging speed
f_old = 0;                                      % Last foraging motion
e = 0.09;                                       % is a positive small number to avoid sigularity
pop_size = 745;                                  % populatin size One Month
%pop_size = 1417;                                  % populatin size Two Months
%% starting krill optimization
K = NaN(pop_size,1);            % vector to store fitness values
R = NaN(pop_size,1); 
D1 = length(ub);             % dimension size
%P = NaN(Np,D)



%krill = data2(:,5:end)
krill = repmat(lb,pop_size,1) + repmat((ub-lb),pop_size,1).*rand(pop_size,D1);  % Generating initial population
lo_best_krill = krill;                                      %local best skrill



for i= 1:pop_size
    R(i) = prob(krill(i,:), data2(i,1:4));       % objective function evaluation
end

for q = 1:pop_size
    K(q) = prob1(R(q));
end
    

K_best = min(K);                                    % best krill fitness value
[val,indexx] = min(K);
index = indexx;
best_kril = krill(index,:);                             % best krill solution

K_worst = max(K);                                       % worst krill fitness value
Ks = NaN(pop_size ,1);
Xi = NaN(pop_size ,D1);
Kss = NaN(pop_size ,1);
Xii = NaN(pop_size ,D1);
alpha_locs = NaN(pop_size ,D1);
comp = NaN(pop_size ,D1);
comp1 = NaN(pop_size ,1);
dts = NaN(1,D1);
p1 = NaN(pop_size ,1);
p2 = NaN(pop_size ,1);
pp1 = NaN(pop_size ,1);
pp2 = NaN(pop_size ,1);
krillnewww = NaN(pop_size, D1);
krillnewww1 = NaN(pop_size, D1);
new_K = NaN(pop_size,1);
global_best_fitness_val = NaN(max_iter ,1);
krillnewww = NaN(pop_size, D1);
krillnewww1 = NaN(pop_size, D1);
new_K = NaN(pop_size,1);
best_krilss = NaN(pop_size, D1);

for u = 1 :max_iter
    for i = 1:pop_size
        D1 = length(ub);
        Ks = NaN(pop_size ,1);
        Xi = NaN(pop_size ,D1);
        Kss = NaN(pop_size ,1);
        Xii = NaN(pop_size ,D1);
        alpha_locs = NaN(pop_size ,D1);
        comp = NaN(pop_size ,D1);
        comp1 = NaN(pop_size ,1);
        dts = NaN(1,D1);
        p1 = NaN(pop_size ,1);
        p2 = NaN(pop_size ,1);
        p3 = NaN(pop_size ,1);
        p4 = NaN(pop_size ,1);
        pp1 = NaN(pop_size ,1);
        pp2 = NaN(pop_size ,1);
        pp3 = NaN(pop_size ,1);
        pp4 = NaN(pop_size ,1);
       
        for j = 1:pop_size
            
            if i ~= j

                Ki = K(i) - K(j)/K_worst - K_best;
                X = (krill(j,:) - krill(i,:))/((norm(krill(j,:) - krill(i,:))) + e);
                
                Ks(j) = Ki;
                
                Xi(j,:)= X;
        Kss = Ks(2:end,:);
        Xii = Xi(2:end,:);
            
            end
        end
        for w = 1 : pop_size
            alpha_loc = Ks(w) * Xi(w,:);         %local effect provided by the neighbors
            alpha_locs(w,:) = alpha_loc;
        end
        
        alpha_locss = sum(alpha_locs, 'omitnan');
        C_best = 2 * ((rand()) + (u/max_iter));         %s the effective coefficient of the krill individual with the best fitness to the ith krill individual
        Ki_best = (K(i) - K_best)/(K_worst - K_best);
        Xi_best = (best_kril - krill(i,:))/((norm(best_kril - krill(i,:))) + e);
        alpha_tag = C_best * Ki_best * Xi_best;  % target vector of each krill individual is the lowest fitness of an individual krill.
        alpha = alpha_locss + alpha_tag; % The motion of krill induced by krill swamp
        Nnew = Nmax * alpha + rand() * Nold;
        
        %% foraging
        % compute food center
        for m = 1:pop_size
            compu = (1/K(m)) * krill(m,:);              % Note compu and compu1 are variables used to store computations
            compu1 = (1/K(m));
            comp(m,:) = compu;
            comp1(m) = compu1;
        end
        X_food = (sum(comp))/(sum(comp1));
        K_food_ =  prob(X_food, data2(i,1:4));  % computing the objective function value of the X_ food
        K_food =  prob1(K_food_);
        
        Ki_food = (K(i) - K_food)/(K_worst - K_best);
        Xi_food = (X_food - krill(i,:))/((norm(X_food - krill(i,:))) + e);
        
        C_food = 2 * (1 - (u/max_iter));    % is the food coefficient
        Beta_food = C_food * Ki_food * Xi_food;   % the food attraction for the ith krill individual
        pb_Ki_ = prob(lo_best_krill(i,:), data2(i,1:4));
        pb_Ki = prob1(pb_Ki_);
        Ki_ibest = (K(i) - pb_Ki )/(K_worst - K_best);   %e best previously visited position of the ith krill individual
        Xi_ibest = (lo_best_krill(i,:) - krill(i,:))/((norm(lo_best_krill(i,:) - krill(i,:))) + e);
        Beta_best = Ki_ibest * Xi_ibest;   %effect of the best fitness of the ith krill individual 
        Beta = Beta_food + Beta_best;
        F = vf * Beta .* rand() .* f_old;
        
        %% compute diffusion

        D_max = 0.002 + rand *(0.010-0.002);        
        gama = -1 + rand *(1-(-1));                 
        D = D_max * (1 - (u/max_iter)) * gama;
        Dx_t = Nnew + F + D;
        ct = 0 + rand *(2-0);                                         
        for l = 1: length(ub)
            dtt = bounds(2,l) - bounds(1,l);                   
            dts(l) = dtt;
        dt = ct * sum(dts);
        X_t_plus_dt = krill(i,:) + dt * Dx_t;
        end
        Cr = 0.2 * Ki_best;
        for h = 1:length(ub)
            for o = 1 : pop_size
                if h == 1
                    
                    p1(o) = krill(o,h);
                end
            end
            for t = 1 : pop_size
                if h == 2
                    p2(t) = krill(t,h);
                end
            end
            for w = 1 : pop_size
                if h == 3
                    p3(w) = krill(w,h);
                end
            end
            for v = 1 : pop_size
                if h == 4
                    p4(v) = krill(v,h);
                end
            end
        end
        p11 = randsample(p1,1);      %random.choice(p1)
        p22 = randsample(p2,1);                      %random.choice(p2)
        p33 = randsample(p3,1);
        p44 = randsample(p4,1);
        p = [p11,p22,p33,p44];
        if rand() < Cr
            X_t_plus_dt = p;
        else
             X_t_plus_dt =  X_t_plus_dt;
        end
        
        %% performing Mutation
        Mu = 0.05/Ki_best;
        mye = rand();
        for w = 1 : length(ub)
            for f = 1 : pop_size
                if w == 1
                    
                    pp1(f) = krill(f,w);
                end
            end
            for c = 1: pop_size
                if w == 2
                    pp2(c) = krill(c,w);
                end
            end
            
            for q = 1: pop_size
                if w == 3
                    pp3(q) = krill(q,w);
                end
            end
            for m = 1: pop_size
                if w == 4
                    pp4(m) = krill(m,w);
                end
            end
        end
        p1x = randsample(pp1,1);
        p1xx = randsample(pp1,1);

        p2x = randsample(pp2,1);
        p2xx = randsample(pp2,1);
        
        p3x = randsample(pp3,1);
        p3xx = randsample(pp3,1);
        
        p4x = randsample(pp4,1);
        p4xx = randsample(pp4,1);
        
        for n = 1 : length(ub)
            if n == 1
                X1 = best_kril(n) + mye * (p1x - p1xx);
            end
            if n == 2
                X2 = best_kril(n) + mye * (p2x - p2xx);
            end
            if n == 3
                X3 = best_kril(n) + mye * (p3x - p3xx);
            end
            if n == 4
                X4 = best_kril(n) + mye * (p4x - p4xx);
            end
        end
        
        Pm = [X1,X2,X3,X4];
        if rand() < Mu
            X_t_plus_dt = Pm;
        else
            X_t_plus_dt = X_t_plus_dt;
        end
        % krill[i] = X_t_plus_dt
        for k = 1 : length(ub)
            if X_t_plus_dt(k) < bounds(1,k)
                X_t_plus_dt(k) = bounds(1,k);
            end
            if X_t_plus_dt(k) > bounds(2,k)
                X_t_plus_dt(k) = bounds(2,k);
            end
        end
        krillnewww(i,:) = X_t_plus_dt;
        
        p1 = [];
        p2 = [];
        p3 = [];
        p4 = [];
        Pm = [];
        Ks = [];
        Xi = [];
        Kss = [];
        Xii = [];
        comp = [];
        comp1 = [];
        dts = [];
        alpha_locs = [];
        pp1 = [];
        pp2 = [];
        pp3 = [];
        pp4 = [];
        Nold = Nnew;
        f_old = F;
      
    end     % UPDATING LOCAL BEST KRILL
    for b = 1:pop_size
    new_K_(b) = prob(krillnewww(i,:),data2(i,1:4)); %prob(lo_best_krill(i,:), data2(i,1:4));      % function evaluation
    end
    for i = 1:pop_size
        new_K(i) = prob1(new_K_(i));
    end
    new_K = fillmissing(new_K,'constant',1000000000000); 
   
    krillnewww1(:,:) = krillnewww;
    [K_bestt,indexx] = min(new_K);
    if K_bestt < K_best
        K_best = K_bestt;
        best_kril = krillnewww(indexx,:);
        global_best_fitness_val(u) = K_best;
    else
        K_best = K_best;
        best_kril =  best_kril;
        global_best_fitness_val(u) = K_best;
    end
    for q = 1 : pop_size
        if new_K(q) < K(q)
            indexxx = find(new_K==new_K(q));
            newpbp = krillnewww1(indexxx,:);
            lo_best_krill(q,:) = newpbp(1,:);                   %newpbp---reprensents new local besst krill position
        end
        if new_K(q) > K(q)
            lo_best_krill(q) = lo_best_krill(q);                %local best krill
        end
    end
    krillnewww1 = [];
    best_krilss(u,:) = best_kril;
    krillnewww = [];
    disp(['iteration ' num2str(u) ': Best fitness value = ' num2str(global_best_fitness_val(u)) ' Optima_sol = ' num2str(best_kril)])
    %% Temperature Actuator
   % if (eT<=-14&&eT>=-28)
   %     fprintf('Temperature Actuator at state T\n');
   % elseif (eT<=-6&&eT>=-22)
   %     fprintf('Temperature Actuator at state T1\n');
   % elseif (eT<=0&&eT>=-14)
   %     fprintf('Temperature Actuator at state T2\n');
   % elseif (eT<=7&&eT>=-7)
   %     fprintf('Temperature Actuator at state T3\n');
   % elseif (eT<=14&&eT>=0)
   %     fprintf('Temperature Actuator at state T4\n');
   % elseif (eT<=22&&eT>=7)
   %     fprintf('Temperature Actuator at state T5\n');
   % else
   %     fprintf('Temperature Actuator at state T6\n');
   % end   

    %% Humidity Actuator
   % if (eH<=-14&&eH>=-28)
   %     fprintf('Humidity Actuator at state T\n');
   % elseif (eH<=-6&&eH>=-22)
   %     fprintf('Humidity Actuator at state T1\n');
   % elseif (eH<=0&&eH>=-14)
   %     fprintf('Humidity Actuator at state T2\n');
   % elseif (eH<=7&&eH>=-7)
   %     fprintf('Humidity Actuator at state T3\n');
   % elseif (eH<=14&&eH>=0)
   %     fprintf('Humidity Actuator at state T4\n');
   % elseif (eH<=22&&eH>=7)
   %     fprintf('Humidity Actuator at state T5\n');
   % else
   %     fprintf('Humidity Actuator at state T6\n');
   % end   
    %% Illumination Actuator

   % if (eL<=-90&&eL>=-175)
   %     fprintf('Illumination Actuator at state I\n');
   % elseif (eL<=-40&&eL>=-120)
   %     fprintf('Illumination Actuator at state I1\n');
   % elseif (eL<=0&&eL>=-90)
   %     fprintf('Illumination Actuator at state I3\n');
   % elseif (eL<=40&&eL>=-40)
   %     fprintf('Illumination Actuator at state I4\n');
   % elseif (eL<=90&&eL>=0)
   %     fprintf('Illumination Actuator at state I5\n');
   % else
   %     fprintf('Illumination Actuator at state I6\n');
   % end   


    %% Ventilation Actuator
   % if (eA<=60&&eA>=-300)
   %     fprintf('Ventilation Actuator at state A\n');
   % elseif (eA>=0&&eA<=120)
   %     fprintf('Ventilation Actuator at state A1\n');
   % elseif (eA<=180&&eA>=60)
   %     fprintf('Ventilation Actuator at state A3\n');
   % elseif (eA>=120&&eA<=240)
   %     fprintf('Ventilation Actuator at state A4\n');
   % else
   %     fprintf('Ventilation Actuator at state A6\n');
   % end
    % dlmwrite('T1.txt',GlobalMax,'-append','delimiter','\n');
    %iter=iter+1;
end % End of KH Opti

%end
[global_bestfitness_val, idex] = min(global_best_fitness_val);
best_sol = best_krilss(idex,:);

disp(['Max objective function value: ' num2str(global_bestfitness_val) ' Optima_sol = ' num2str(best_sol)])

plot(1:max_iter,global_best_fitness_val)
xlabel('Iteration');
ylabel('Objective function value');
title('convergance plot');

save all               
       







