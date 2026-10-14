% Author: Grace Calkins
% Date: 08/15/2025
% Purpose: Symmetrize all HOCGTs, sCGTs, qCGTs for the aerocapture trajectory 
% and solve higher-order tensor eigenpair problem with SSHOPM solver and
% save off maximal eigenpairs

close all;  clear all;  clc;

parallelFlag = true;
numWorkers = 10;   
 

% Journal 
% dataPath = "/Users/gracecalkins/Local_Documents/local_code/sttgmm/data/JP_dstts_nominal";
% figPath = "/Users/gracecalkins/Local_Documents/local_code/sttgmm/figures/JP_dstts_nominal";
% saveTag = 'JP_dstts_nominal';

% dataPath = "/Users/gracecalkins/Local_Documents/local_code/sttgmm/data/JP_dstts_nominal_drag_mod";
% figPath = "/Users/gracecalkins/Local_Documents/local_code/sttgmm/figures/JP_dstts_nominal_drag_mod";
% saveTag = 'JP_dstts_nominal_drag_mod';

dataPath = "/Users/gracecalkins/Local_Documents/local_code/sttgmm/data/JP_dstts_nominal_fast";
figPath = "/Users/gracecalkins/Local_Documents/local_code/sttgmm/figures/JP_dstts_nominal_fast";
saveTag = 'JP_dstts_nominal_fast';

% OVERALL
all_regimes = {}; % rows = HOCGT type, cols = regimes

% all_regimes{1,1} = 0:1:154; % HOCGTs from t0, tk
% all_regimes{1,2} = 0:1:154; % HOCGTs from tk, tk+1
% all_regimes{1,3} = 0:1:155; % HOCGTs from tk, tf
% % all_regimes{1,4} = [0,1,2]; % HOCGTs from regime
% all_regimes{1,5} = [0]; % HOCGTs from t0, tf
% 
% all_regimes{2,1} = 0:1:154; % SCGTs from t0, tk
% all_regimes{2,2} = 0:1:154; % SCGTs from tk, tk+1
% all_regimes{2,3} = 0:1:155; % SCGTs from tk, tf
% % all_regimes{2,4} = [0,1,2]; % SCGTs from regime
% all_regimes{2,5} = [0]; % SCGTs from t0, tf
% 
% all_regimes{3,1} = 0:1:154; % energy qCGTs from t0, tk
% all_regimes{3,2} = 0:1:154; % energy qCGTs from tk, tk+1
% all_regimes{3,3} = 0:1:155; % energy qCGTs from tk, tf
% % all_regimes{3,4} = [0,1,2]; % energy qCGTs from regime
% all_regimes{3,5} = [0]; % energy qCGTs from t0, tf
% 
% all_regimes{4,3} = 0:1:155; % apoapsis qCGTs from tk, tf
% all_regimes{4,5} = [0]; % apoapsis qCGTs from t0, tf
% 
% prefixes = ["HOCGT", "sCGT", "energy_qCGT", "apoRad_qCGT"];
% regimeTags = ["t0tk", "tktkp1", "tktf", "regime", "t0tf"];

% Subselections
prefixes = ["energy_qCGT", "HOCGT", "sCGT"];
% prefixes = ["pos_qCGT", "rOnly_sCGT"];
% prefixes = ["energy_qCGT", "apoRad_qCGT"];
regimeTags = ["t0tk"]; 
% regimeTags = [ "tktf"]; 

all_regimes{1,1} = 0:1:154; % HOCGTs from t0, tk
% all_regimes{1,1} = 50:1:100; % HOCGTs from t0, tk
% all_regimes{1,2} = 0:1:154; % HOCGTs from tk, tk+1
% all_regimes{1,1} = 0:1:154; % HOCGTs from tk, tf
% all_regimes{1,1} = [0,1,2]; % HOCGTs from regime
% all_regimes{1,3} = [0]; % HOCGTs from t0, tf

all_regimes{2,1} = 0:1:154; % SCGTs from t0, tk
% all_regimes{2,2} = 0:1:154; % SCGTs from tk, tk+1
% all_regimes{2,1} = 0:1:154; % SCGTs from tk, tf
% all_regimes{2,2} = [0,1,2]; % SCGTs from regime
% all_regimes{2,3} = [0]; % SCGTs from t0, tf

all_regimes{3,1} = 0:1:154; % energy qCGTs from t0, tk
% all_regimes{3,2} = 0:1:154; % energy qCGTs from tk, tk+1
% all_regimes{3,1} = 0:1:154; % energy qCGTs from tk, tf
% all_regimes{3,2} = [0,1,2]; % energy qCGTs from regime
% all_regimes{3,3} = [0]; % energy qCGTs from t0, tf
% 
% all_regimes{4,1} = 0:1:154; % apoapsis qCGTs from tk, tf
% all_regimes{4,3} = [0]; % apoapsis qCGTs from t0, tf

colors = define_colors();
stateSize = 7;
maxOrder = 3; 

%% Loop
overall_start = tic;

if parallelFlag
    % Start or reuse a pool
    if isempty(gcp('nocreate'))
        parpool('local', numWorkers);
    else
        p = gcp('nocreate');
        if p.NumWorkers ~= numWorkers
            delete(gcp('nocreate'));      % close current pool
            parpool('local', numWorkers); % start new one
        end
    end

    % Precompute all combinations of (ii,jj) to avoid nesting inside parfor
    tasks = {};
    for ii = 1:length(prefixes)
        for jj = 1:length(regimeTags)
            if ~isempty(all_regimes{ii,jj})
                tasks{end+1} = {ii, jj}; 
            end
        end
    end
    
    parfor kk = 1:numel(tasks)
        ii = tasks{kk}{1};
        jj = tasks{kk}{2};
        prefix = prefixes(ii);
        regimes = all_regimes{ii,jj};
        regimeTag = regimeTags(jj);

        fprintf(strcat(prefix, " ", regimeTag, "\n"));
        HOCGTS_nonsym_all = load_nonsymmetric_tensors(dataPath, regimes, strcat(prefix, "_", regimeTag), 'HOCGT', saveTag, maxOrder);
        HOCGTs_sym_all = symmetrize_tensors(HOCGTS_nonsym_all, maxOrder);
        save_sym_tensors(HOCGTS_nonsym_all, dataPath, regimes, strcat("SYM_", prefix, "_", regimeTag), 'HOCGT', maxOrder)
        [ho_lambdas, ho_evecs, ho_flags, ho_all_lambdas, ho_all_evecs] = compute_eigenpairs_multiguess(HOCGTs_sym_all, stateSize, maxOrder, false);
        % plot_eigenvalues(ho_lambdas, regimes, colors, figPath, strcat("hoevals_", prefix, "_", regimeTag), maxOrder, ho_flags);
        % plot_eigenvectors(ho_evecs, regimes, stateSize, colors, figPath, strcat("hoevecs_", prefix, "_", regimeTag), maxOrder, ho_flags);
        % plot_unique_eigenvalues(maxOrder, regimes, ho_all_lambdas, figPath, saveTag, regimeTag)
        % plot_n_ho_evals(2, maxOrder, regimes, ho_all_lambdas, figPath, saveTag, regimeTag, colors)
        % plot_n_ho_evecs(2, maxOrder, regimes, ho_all_evecs, figPath, saveTag, regimeTag, colors, stateSize)
        save_eigenvectors(ho_evecs, dataPath, 'ho', strcat(prefix, '_', regimeTag))
        save_eigenvalues(ho_lambdas, dataPath, 'ho', strcat(prefix, '_', regimeTag))
        save_HOeigenvectors(ho_all_evecs, dataPath, 'ho', strcat(prefix, '_', regimeTag))
        save_HOeigenvalues(ho_all_lambdas, dataPath, 'ho', strcat(prefix, '_', regimeTag))
        % % close all;
    end

else
    for ii = 1:length(prefixes)  % loop over types of HOCGTs
        prefix = prefixes(ii);
            
        for jj = 1:length(regimeTags) % loop over regime types
            if ~isempty(all_regimes{ii, jj})
                regimes = all_regimes{ii, jj};
                regimeTag = regimeTags(jj);
    
                fprintf(strcat(prefix, " ", regimeTag, "\n"))
        
                HOCGTS_nonsym_all = load_nonsymmetric_tensors(dataPath, regimes, strcat(prefix, "_", regimeTag), 'HOCGT', saveTag, maxOrder);
                HOCGTs_sym_all = symmetrize_tensors(HOCGTS_nonsym_all, maxOrder);
                save_sym_tensors(HOCGTS_nonsym_all, dataPath, regimes, strcat("SYM_", prefix, "_", regimeTag), 'HOCGT', maxOrder)
                [ho_lambdas, ho_evecs, ho_flags, ho_all_lambdas, ho_all_evecs] = compute_eigenpairs_multiguess(HOCGTs_sym_all, stateSize, maxOrder, false);
                plot_eigenvalues(ho_lambdas, regimes, colors, figPath, strcat("hoevals_", prefix, "_", regimeTag), maxOrder, ho_flags);
                plot_eigenvectors(ho_evecs, regimes, stateSize, colors, figPath, strcat("hoevecs_", prefix, "_", regimeTag), maxOrder, ho_flags);
                plot_unique_eigenvalues(maxOrder, regimes, ho_all_lambdas, figPath, saveTag, regimeTag)
                plot_n_ho_evals(2, maxOrder, regimes, ho_all_lambdas, figPath, saveTag, regimeTag, colors)
                plot_n_ho_evecs(2, maxOrder, regimes, ho_all_evecs, figPath, saveTag, regimeTag, colors, stateSize)
                save_eigenvectors(ho_evecs, dataPath, 'ho', strcat(prefix, '_', regimeTag))
                save_eigenvalues(ho_lambdas, dataPath, 'ho', strcat(prefix, '_', regimeTag))
                save_HOeigenvectors(ho_all_evecs, dataPath, 'ho', strcat(prefix, '_', regimeTag))
                save_HOeigenvalues(ho_all_lambdas, dataPath, 'ho', strcat(prefix, '_', regimeTag))
                close all;
            end
        
        end
    end
end
overall_time = toc(overall_start);
fprintf('Overall time: %.2f seconds (%.2f minutes)\n', overall_time, overall_time/60);


%% Functions
function colors = define_colors()
    colors = [
        102/255, 194/255, 165/255;
        252/255, 141/255, 98/255;
        141/255, 160/255, 203/255;
        231/255, 138/255, 195/255;
        166/255, 216/255, 84/255;
        255/255, 217/255, 47/255;
        229/255, 196/255, 148/255;
        179/255, 179/255, 179/255
    ];
end

function windows = generate_windows(ts, Nskip, win_len)
    index = 1;
    windows = [];
    for ii = 1:Nskip:length(ts)-1
        if ts(ii) + win_len < ts(end)
            window = [ts(ii), ts(ii)+win_len];
        else
            window = [ts(ii), ts(end)];
        end
        windows(index, :) = window;
        index = index + 1;
    end
    windows = windows(1:end-1, :);
end

function tensors_all = load_nonsymmetric_tensors(dataPath, regimes, filePrefix, tensorPrefix, saveTag, maxOrder)
    tensors_all = {};
    for jj = 1:length(regimes)  % Regime
        tensors = {};
        for ii = 1:maxOrder  % Order
            tensorData = load(fullfile(dataPath, filePrefix + "_" + string(regimes(jj)) + "_O" + string(ii) + "_" + saveTag + ".mat"));
            tensors{ii} = tensorData.(tensorPrefix + string(ii-1));
        end
        tensors_all{jj} = tensors;
    end
end

function tensors_all = load_nonsymmetric_tensors_t0tf(dataPath, regimes, filePrefix, tensorPrefix, saveTag, maxOrder)
    tensors_all = {};
    for jj = 1:length(regimes)  % Regime
        tensors = {};
        for ii = 1:maxOrder  % Order
            tensorData = load(fullfile(dataPath, filePrefix + "_t0_tf_O" + string(ii) + "_" + saveTag + ".mat"));
            tensors{ii} = tensorData.(tensorPrefix + string(ii-1));
        end
        tensors_all{jj} = tensors;
    end
end

function sym_tensors_all = symmetrize_tensors(tensors_all, maxOrder)
    sym_tensors_all = {};
    for jj = 1:length(tensors_all)
        sym_tensors = {};
        for ii = 1:maxOrder
            sym_tensors{ii} = symmetrize(tensor(tensors_all{jj}{ii}));
        end
        sym_tensors_all{jj} = sym_tensors;
    end
end
 
function save_sym_tensors(sym_tensors, dataPath, regimes, filePrefix, tensorPrefix, maxOrder)
    for jj = 1:length(regimes)
        tensors = sym_tensors(jj);
        for ii = 1:maxOrder
            tensor = struct('tensor', tensors{1}{ii});
            save(fullfile(dataPath, filePrefix + "_O" + string(ii-1) + "_R" + string(regimes(jj)) + ".mat"), 'tensor');
        end
    end
end


function [lambdas, evecs, flags] = compute_eigenpairs(tensors_all, stateSize, maxOrder)
    num_windows = length(tensors_all);
    lambdas = zeros(maxOrder, num_windows);
    flags = zeros(maxOrder, num_windows);
    evecs = zeros(maxOrder, num_windows, stateSize); 
    maxGuessIter = 10; % Maximum number of random initial guesses to try if the first one does not converge

    for jj = 1:num_windows
        for ii = 1:maxOrder
            % Compute shift parameter as sum of absolute value of all tensor
            % entries * (stateSize - 1) from eqn 26 & 27 of
            % "LINEAR COVARIANCE FIDELITY CHECKS AND MEASURES OF NON-GAUSSIANITY" by Kulik et al 
            % TODO - apative shift in the function does this automatically
            % shift_param = (stateSize - 1)*sum(abs(double(tensors_all{jj}{ii})), 'all'); 
            % disp("Shift param for window " + string(jj) + " and order " + string(ii) + " is " + string(shift_param))
            flag = -2; guessIter = 0;
            
    
            while flag ~= 0 && guessIter <= maxGuessIter
                disp("Window " + string(jj) + ", order " + string(ii+1) + "; guess iter " + string(guessIter))
    
                if guessIter == 0 && ii > 1  % Use lower order tensor eigenpair as initial guess
                    disp("-> Using lower order solution as initial guess")
                    [lambda, evec, flag] = eig_sshopm(tensors_all{jj}{ii}, 'Shift', 'adaptive', 'Start', squeeze(evecs(ii-1, jj, :)));
                elseif ii == 1 % for first order just solve a regular eigenvalue problem 
                    CGT = tenmat(tensors_all{jj}{ii}, 1).data;  % map tensor object to matrix object
                    [vecs, evals] = eig(CGT);
                    [lambda, maxind] = max(diag(evals));
                    evec = vecs(:, maxind);
                    flag = 0;
                else  % random initial guess
                    [lambda, evec, flag] = eig_sshopm(tensors_all{jj}{ii}, 'Shift', 'adaptive');
                end
                    
                disp("  -> Eval solver flag: " + string(flag) + " (0 = success, -1 = norm(x) = 0, -2 = max iters exceeded)")
                guessIter = guessIter + 1;
            end
            if guessIter == maxGuessIter + 1
                disp("!! !! !! !! ALL INITIAL GUESSES FAILED !! !! !! !! ")
            end

            lambdas(ii, jj) = lambda;
            evecs(ii, jj, :) = (evec);
            flags(ii, jj) = flag;
        end
    end
end

function [lambdas, evecs, flags, all_lambdas, all_evecs] = compute_eigenpairs_multiguess(tensors_all, stateSize, maxOrder, verbFlag)
    num_windows = length(tensors_all);
    lambdas = zeros(maxOrder, num_windows);
    all_lambdas = {};
    all_evecs = {};
    flags = zeros(maxOrder, num_windows);
    evecs = zeros(maxOrder, num_windows, stateSize); 

    N = 100; % TODO - trying to get less discontinuity for the HOCGTs and sCGTs  % Number of samples to guess 
    angle_tol = 1e-2;

    totalIter = num_windows * maxOrder;
    iterCount = 0;
    tic; % start timer
    h = waitbar(0,'Starting...');

    for jj = 1:num_windows
        for ii = 1:maxOrder
            iterCount = iterCount + 1;

            % progress
            frac = iterCount / totalIter;

            % timing
            elapsed = toc;
            est_total = elapsed / frac;
            remaining = est_total - elapsed;

            % print on one line
            waitbar(frac, h, sprintf(' %3.1f%%  Elapsed: %5.1fs  Rem: %5.1fs', frac*100, elapsed, remaining));
            % fprintf('\r%s %5.1f%% | Elapsed: %6.1fs | Remaining: %6.1fs', ...
                % barStr, percent, elapsed, remaining); 

            if verbFlag
                disp("Window " + string(jj) + ", order " + string(ii+1))
            end

            if ii > 1 % For higher orders, check a bunch of eigenpairs
                lambda_list = zeros(N,1);
                x_list = cell(N,1);

                % Run SS-HOPM from multiple initial vectors
                for kk = 1:N
                    x0 = randn(stateSize,1);
                    x0 = x0 / norm(x0);
                    shift_param = (stateSize - 1) * sum(abs(double(tensors_all{jj}{ii})), 'all');
                    [lambda, x, flag] = eig_sshopm(tensors_all{jj}{ii}, 'Start', x0, 'Shift', 'Adaptive', 'MaxIts', 5000, 'Tol', 1e-6); %, 'Display', 1000);
                    if flag == 0  % successfully converged
                        lambda_list(kk) = lambda;
                        x_list{kk} = x;
                    else
                        lambda_list(kk) = -10000;
                        x_list{kk} = zeros(length(x),1);
                    end
                end

                % Sort by eigenvalue (descending)
                [sorted_lambda, idx] = sort(lambda_list, 'descend');
                sorted_x = x_list(idx);

                % Deduplicate based on angular similarity
                unique_x = {};
                unique_lambda = [];

                lambda_tol = 1e-6; % tolerance for treating eigenvalues as identical

                for i = 1:N
                    candidate_x = sorted_x{i};
                    candidate_lambda = sorted_lambda(i);
                
                    is_duplicate = false;
                    same_lambda_indices = find(abs(candidate_lambda - unique_lambda) < lambda_tol);
                
                    % Check angular similarity only within same-eigenvalue group
                    for j = same_lambda_indices
                        dot_product = abs(dot(candidate_x, unique_x{j}));
                        if dot_product > cos(angle_tol)
                            is_duplicate = true;
                            break;
                        end
                    end
                
                    if ~is_duplicate
                        unique_x{end+1} = candidate_x;
                        unique_lambda(end+1) = candidate_lambda;
                
                        % Check if there are now multiple distinct eigenvectors for this eigenvalue
                        same_lambda_count = sum(abs(candidate_lambda - unique_lambda) < lambda_tol);
                        if same_lambda_count > 1
                            fprintf('Multiple distinct eigenvectors found for eigenvalue %.6g (count = %d)\n', ...
                                    candidate_lambda, same_lambda_count);
                        end
                    end
                end
                
                % for i = 1:N
                %     candidate_x = sorted_x{i};
                %     is_duplicate = false;
                % 
                %     for j = 1:length(unique_x)
                %         dot_product = abs(dot(candidate_x, unique_x{j}));
                %         if dot_product > cos(angle_tol)
                %             is_duplicate = true;
                %             break;
                %         end
                %     end
                % 
                %     if ~is_duplicate
                %         unique_x{end+1} = candidate_x; 
                %         unique_lambda(end+1) = sorted_lambda(i); 
                %     end
                % end

                % Resort the unique lambdas and get the largest eigenpair
                [sorted_lambda_dedupe, idx] = sort(unique_lambda, 'descend');
                sorted_x_dedupe = unique_x(idx);

                evec = cell2mat(sorted_x_dedupe(:,1));
                lambda = sorted_lambda_dedupe(1);
                flag = 0;

                % % Plotting for debugging
                % plot_evecs = zeros(N, stateSize);
                % for ll = 1:length(sorted_x)
                %     plot_evecs(ll, :) = sorted_x{ll};
                % end
                % figure;
                % subplot(2,1,1)
                % plot(sorted_lambda, linewidth=2)
                % ylabel('Eigenvalue Magnitude')
                % subplot(2,1,2); hold on;
                % for ll = 1:stateSize
                %     plot(plot_evecs(:,ll), linewidth=2)
                % end
                % ylabel('Eigenvector Coordinates')
                % xlabel('Guess Index')
                % legend({'r', '\theta', '\phi', 'V', '\gamma', '\psi', 'ln \rho'})
                % sgtitle(string(length(unique_lambda)) + " unique eigenpairs")


                % Display the results
                if verbFlag
                    fprintf('Found %d unique eigenpairs out of %d trials.\n', length(unique_lambda), N);
                    fprintf('Top eigenvalue: %.10f\n', unique_lambda(1));
                end
                            
            elseif ii == 1 % for first order just solve a regular eigenvalue problem 
                CGT = tenmat(tensors_all{jj}{ii}, 1).data;  % map tensor object to matrix object
                [vecs, evals] = eig(CGT);
                [lambda, maxind] = max(diag(evals));
                evec = vecs(:, maxind);
                % Sort eigenvalues in descending order (along with indices)
                [sorted_lambda, idx] = sort(diag(evals), 'descend');
                flag = 0;
                sorted_lambda_dedupe = diag(sorted_lambda);
                sorted_x_dedupe = vecs(:, idx);
            else  % random initial guess
                [lambda, evec, flag] = eig_sshopm(tensors_all{jj}{ii}, 'Shift', 'adaptive');
            end
                
            if verbFlag
                disp("  -> Eval solver flag: " + string(flag) + " (0 = success, -1 = norm(x) = 0, -2 = max iters exceeded)")
            end


            all_lambdas{ii, jj} = sorted_lambda_dedupe;
            all_evecs{ii, jj} = sorted_x_dedupe;
            lambdas(ii, jj) = lambda;
            evecs(ii, jj, :) = (evec);
            flags(ii, jj) = flag;
        end
    end
    close(h);
end

function plot_eigenvalues(lambdas, regimes, colors, figPath, filename, maxOrder, flags)
    figure;
    hold on; grid on;

    % Preallocate line handles for legend
    legendHandles = gobjects(maxOrder, 1);

    for ii = 1:maxOrder
        % Plot the connecting line and store handle for legend
        legendHandles(ii) = plot(regimes, abs(lambdas(ii, :)), ...
            'Color', colors(ii, :), ...
            'LineWidth', 2);

        % Plot each marker individually based on the flag
        for jj = 1:length(regimes)
            if flags(ii, jj)~= 0
                markerFaceColor = colors(ii, :);  % filled
                plot(regimes(jj), abs(lambdas(ii, jj)), ...
                    'o', ...
                    'Color', colors(ii, :), ...
                    'MarkerFaceColor', markerFaceColor, ...
                    'MarkerSize', 10, ...
                    'LineWidth', 1.5);
            end
        end
    end

    set(gca, 'YScale', 'log');
    xlim([-inf, inf])
    xlabel("Regime Number");
    ylabel("Eigenvalue Magnitude");
    sgtitle("Eigenvalue Magnitudes over Time");

    % Create legend using only the line handles
    legend(legendHandles, ["Order 2", "Order 3", "Order 4"], 'Location', 'best');

    exportgraphics(gcf, fullfile(figPath, filename + ".png"), 'Resolution', 600);
end


function plot_eigenvectors(evecs, regimes, stateSize, colors, figPath, filename, maxOrder, flags)
    figure;
    x_labels = {'r', '\theta', '\phi', 'V', '\gamma', '\psi', 'ln \rho'};
    
    % Store line handles for legend
    legendHandles = gobjects(stateSize, 1);

    for ii = 1:maxOrder
        ax = subplot(maxOrder, 1, ii); hold(ax, 'on'); grid(ax, 'on');

        for jj = 1:stateSize
            % Plot the solid line and store handle only in the first subplot
            h = plot(regimes, abs(evecs(ii, :, jj)), ...
                'Color', colors(jj, :), ...
                'LineWidth', 2);

            if ii == 1
                legendHandles(jj) = h;
            end

            % Plot individual markers based on flags
            for kk = 1:length(regimes)
                if flags(ii, kk) ~= 0
                    markerFaceColor = colors(jj, :);  % filled
                    plot(regimes(kk), abs(evecs(ii, kk, jj)), ...
                        'o', ...
                        'Color', colors(jj, :), ...
                        'MarkerFaceColor', markerFaceColor, ...
                        'MarkerSize', 8, ...
                        'LineWidth', 1.2);
                end
            end
        end

        ylabel("Order " + string(ii + 1));
        ylim([0, 1]);
        xlim([-inf, inf])
    end

    subplot(maxOrder, 1, maxOrder);
    xlabel("Regime Number");

    sgtitle("Eigenvector Magnitudes");

    % Put legend in the first subplot, using only line handles
    subplot(maxOrder, 1, 1);
    legend(legendHandles, x_labels, 'Location', 'northeast');

    set(gcf, 'Position', [0, 0, 700, 800]);
    exportgraphics(gcf, fullfile(figPath, filename + ".png"), 'Resolution', 600);
end



function save_eigenvectors(evecs, dataPath, flag, runTag)
    if flag == 's'
        % s_Rs = assign_to_ts(ts, windows, evecs);
        s_Rs = evecs;
        save(fullfile(dataPath, string(runTag) + '_s_Rs.mat'), 's_Rs');
    elseif flag == 'q'
        % q_Rs = assign_to_ts(ts, windows, evecs);
        q_Rs = evecs;
        save(fullfile(dataPath, string(runTag) + '_q_Rs.mat'), 'q_Rs');
    else
        % ho_Rs = assign_to_ts(ts, windows, evecs);
        ho_Rs = evecs;
        save(fullfile(dataPath, string(runTag) + '_ho_Rs.mat'), 'ho_Rs');
    end
end

function save_eigenvalues(evals, dataPath, flag, runTag)
    if flag == 's'
        s_evals = evals;
        save(fullfile(dataPath, string(runTag) + '_s_evals.mat'), 's_evals');
    elseif flag == 'q'
        q_evals = evals;
        save(fullfile(dataPath, string(runTag) + '_q_evals.mat'), 'q_evals');
    else
        ho_evals = evals;
        save(fullfile(dataPath, string(runTag) + '_ho_evals.mat'), 'ho_evals');
    end
end

function save_HOeigenvectors(ho_evecs, dataPath, flag, runTag)
    if flag == 's'
        % s_Rs = assign_to_ts(ts, windows, evecs);
        s_Rs = ho_evecs;
        save(fullfile(dataPath, string(runTag) + '_HOs_Rs.mat'), 's_Rs');
    elseif flag == 'q'
        % q_Rs = assign_to_ts(ts, windows, evecs);
        q_Rs = ho_evecs;
        save(fullfile(dataPath, string(runTag) + '_HOq_Rs.mat'), 'q_Rs');
    else
        % ho_Rs = assign_to_ts(ts, windows, evecs);
        ho_Rs = ho_evecs;
        save(fullfile(dataPath, string(runTag) + '_HOho_Rs.mat'), 'ho_Rs');
    end
end

function save_HOeigenvalues(ho_evals, dataPath, flag, runTag)
    if flag == 's'
        s_evals = ho_evals;
        save(fullfile(dataPath, string(runTag) + '_HOs_evals.mat'), 's_evals');
    elseif flag == 'q'
        q_evals = ho_evals;
        save(fullfile(dataPath, string(runTag) + '_HOq_evals.mat'), 'q_evals');
    else
        ho_evals = ho_evals;
        save(fullfile(dataPath, string(runTag) + '_HOho_evals.mat'), 'ho_evals');
    end
end


function Rs = assign_to_ts(ts, windows, evecs)
    Rs = zeros(length(ts), size(evecs, 1), size(evecs, 3));
    t_starts = windows(:, 1);
    for ii = 1:length(ts)
        ind = find(ts(ii) >= t_starts, 1, 'last');
        Rs(ii, :, :) = evecs(:, ind, :);
    end
end

function plot_unique_eigenvalues(maxOrder, regimes, ho_all_lambdas, figPath, saveTag, regimeTag)
    % Get number of unique eigenpairs for all windows and orders 
    num_unique_epairs = zeros(maxOrder, length(regimes));
    for oo = 1:maxOrder
        for ww = regimes+1
            num_unique_epairs(oo, ww) = length(ho_all_lambdas{oo, ww});
        end
    end
    
    % Make subplots with a row for each order and on each subplot a bar for the number of unique eigenpairs for that window 
    figure;
    for oo = 1:maxOrder
        subplot(maxOrder, 1, oo);
        bar(regimes, num_unique_epairs(oo, :));
        title(['Order ', num2str(oo)]);
        ylabel('# Eigenpairs');
        if oo == maxOrder
            xlabel('Regime Index');
        end
        grid on;
    end
    sgtitle('Number of Unique Eigenpairs per Regime and Order');
    exportgraphics(gcf, fullfile(figPath, "NUM_HO_evals_"+saveTag+'_'+regimeTag + ".png"), 'Resolution', 600);
end

function plot_n_ho_evals(nPlot, maxOrder, regimes, ho_all_lambdas, figPath, saveTag, regimeTag, colors)
    % Plot top nPlot eigenvalues on a subplot for each order. If there is only
    % 1 eigenvalue in ho_all_lambdas{oo, ww}, leave a blank space in the plot. 
    figure;
    
    for oo = 1:maxOrder
        subplot(maxOrder, 1, oo);
        hold on;
    
        % Initialize matrix to store top nPlot eigenvalues (real part) for all regimes
        top_lambda_real = NaN(nPlot, length(regimes));
    
        for rr = 1:length(regimes)
            ww = regimes(rr)+1;
            lambdas = ho_all_lambdas{oo, ww};
    
            % Always store λ₁ if it exists
            if length(lambdas) >= 1
                top_lambda_real(1, rr) = real(lambdas(1));
            end

            % Store higher eigenvalues if available
            if length(lambdas) > 1
                max_k = min(nPlot, length(lambdas));
                top_lambda_real(2:max_k, rr) = real(lambdas(2:max_k));
            end
    
        end
    
        % Plot each eigenvalue index across regimes
        for kk = 1:nPlot
            semilogy(regimes, top_lambda_real(kk, :), '-o', 'MarkerFaceColor', colors(kk,:), 'DisplayName', ['\lambda_', num2str(kk)], color=colors(kk,:));
        end
    
        title(['Top ', num2str(nPlot), ' Eigenvalues for Order ', num2str(oo)]);
        ylabel('Re(\lambda)');
        if oo == maxOrder
            xlabel('Regime Index');
        end
        xlim([min(regimes)-0.5, max(regimes)+0.5]);
        grid on;
        legend('show');
    end
    
    sgtitle(['Top ', num2str(nPlot), ' Eigenvalues per Regime and Order']);
    exportgraphics(gcf, fullfile(figPath, "TOP_"+nPlot+"_HO_evals_"+saveTag+'_'+regimeTag + ".png"), 'Resolution', 600);
end

function plot_n_ho_evecs(nPlot, maxOrder, regimes, ho_all_evecs, figPath, saveTag, regimeTag, colors, stateSize)
    % Plot top nPlot eigenvectors from ho_all_evecs for each order 
    % if there is only 1 eigenvalue in ho_all_evecs{oo, ww}, leave a blank space in the plot.
    % Make subplots with rows for the orders and columns for each eigenvector
    x_labels = {'r', '\theta', '\phi', 'V', '\gamma', '\psi', 'ln \rho'};
    figure('Position', [100, 100, 1000, 500]);
    
    % Store handles for legend entries
    plot_handles = gobjects(stateSize, 1);
    
    for oo = 1:maxOrder
        for kk = 1:nPlot
            subplot(maxOrder, nPlot, (oo - 1)*nPlot + kk);
            hold on;
    
            % Initialize storage: one row per regime, one column per coordinate
            evec_data = NaN(length(regimes), stateSize);
    
            for rr = 1:length(regimes)
                ww = regimes(rr)+1;
                evecs = ho_all_evecs{oo, ww};
    
                if kk > 1 && (size(evecs, 2) <= 1 || kk > size(evecs, 2))
                    continue;  % Leave NaN for λ₂ and higher if not available
                end

    
                if iscell(evecs)
                    vec = real(evecs{kk});  % Extract from cell and take real part
                else
                    vec = real(evecs(:, kk));  % Directly take column if numeric
                end 
                evec_data(rr, :) = abs(vec(:))';  % Store as row in matrix
            end
    
            % Now plot each coordinate across regimes
            for cc = 1:stateSize
                h = plot(regimes, evec_data(:, cc), '-o', ...
                         'DisplayName', x_labels{cc}, 'MarkerFaceColor', colors(cc,:), color=colors(cc,:));
                if oo == 1 && kk == 1
                    plot_handles(cc) = h;  % Save for legend
                end
            end
    
            title(['Order ', num2str(oo), ', \lambda_', num2str(kk)]);
            xlim([min(regimes)-0.5, max(regimes)+0.5]);
            grid on;
            ylim([0, 1]);
            if oo == maxOrder
                xlabel('Regime');
            end
            if kk == 1
                ylabel('Eigenvector Component');
            end
        end
    end
    % Create legend to the right of all subplots
    legend_ax = axes('Position',[0 0 0.99 1],'Visible','off');  % Invisible axis
    legend(legend_ax, plot_handles, x_labels, ...
           'Location','eastoutside', 'FontSize', 10);
    
    sgtitle(['Top ', num2str(nPlot), ' Eigenvectors Across Regimes']);
    exportgraphics(gcf, fullfile(figPath, "TOP_"+length(nPlot)+"_HO_evecs_"+saveTag+'_'+regimeTag + ".png"), 'Resolution', 600);
end




