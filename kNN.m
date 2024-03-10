
% If using for actual files: do: feature_matrix = load(data_file.txt)
feature_matrix = [
    0.0900, 0.0162, 0.0068, 0.0081, 1.3478, 1.8924, 1.0000; % Channel 1 A01_stress_mat
    0.0795, 0.0136, 0.0058, 0.0072, 1.2143, 1.7971, 1.0000; % Channel 2 A01_stress_mat
    0.0893, 0.0153, 0.0063, 0.0074, 1.2876, 1.8617, 1.0000; % Channel 3 A01_stress_mat
    0.0818, 0.0147, 0.0062, 0.0073, 1.2405, 1.8212, 1.0000; % Channel 4 A01_stress_mat

    0.0424, 0.0082, 0.0035, 0.0040, 0.8027, 1.4624, 1.0000; % Channel 1 A02_stress_mat
    0.0489, 0.0093, 0.0039, 0.0046, 0.8658, 1.5225, 1.0000; % Channel 2 A02_stress_mat
    0.0429, 0.0088, 0.0037, 0.0044, 0.8284, 1.4809, 1.0000; % Channel 3 A02_stress_mat
    0.0643, 0.0133, 0.0055, 0.0066, 1.1077, 1.7141, 1.0000; % Channel 4 A02_stress_mat

    0.0846, 0.0164, 0.0070, 0.0082, 1.3498, 1.8822, 1.0000; % Channel 1 A10_stress_mat
    0.0862, 0.0168, 0.0072, 0.0081, 1.3066, 1.8703, 1.0000; % Channel 2 A10_stress_mat
    0.0881, 0.0169, 0.0072, 0.0085, 1.3314, 1.8834, 1.0000; % Channel 3 A10_stress_mat
    0.0891, 0.0172, 0.0073, 0.0086, 1.3489, 1.8948, 1.0000; % Channel 4 A10_stress_mat

    0.0893, 0.0176, 0.0075, 0.0083, 1.3045, 1.8794, 1.0000; % Channel 1 A11_stress_mat
    0.0880, 0.0174, 0.0074, 0.0082, 1.2867, 1.8660, 1.0000; % Channel 2 A11_stress_mat
    0.0892, 0.0174, 0.0074, 0.0084, 1.3273, 1.8896, 1.0000; % Channel 3 A11_stress_mat
    0.0887, 0.0174, 0.0074, 0.0082, 1.3018, 1.8739, 1.0000; % Channel 4 A11_stress_mat

    0.0862, 0.0165, 0.0071, 0.0079, 1.2477, 1.8361, 1.0000; % Channel 1 A12_stress_mat
    0.0880, 0.0169, 0.0072, 0.0081, 1.2652, 1.8498, 1.0000; % Channel 2 A12_stress_mat
    0.0851, 0.0169, 0.0071, 0.0079, 1.2554, 1.8412, 1.0000; % Channel 3 A12_stress_mat
    0.0815, 0.0159, 0.0069, 0.0077, 1.2540, 1.8296, 1.0000; % Channel 4 A12_stress_mat

    0.0804, 0.0126, 0.0056, 0.0066, 1.2160, 1.7816, 0; % Channel 1 A02_normal_mat
    0.0789, 0.0146, 0.0063, 0.0073, 1.2635, 1.8222, 0; % Channel 2 A02_normal_mat
    0.0924, 0.0173, 0.0071, 0.0085, 1.4066, 1.9236, 0; % Channel 3 A02_normal_mat
    0.0795, 0.0159, 0.0066, 0.0079, 1.2985, 1.8463, 0; % Channel 4 A02_normal_mat

    0.0869, 0.0156, 0.0066, 0.0077, 1.3245, 1.8743, 0; % Channel 1 A02_normal_mat
    0.0883, 0.0172, 0.0074, 0.0083, 1.2917, 1.8676, 0; % Channel 2 A02_normal_mat
    0.0914, 0.0184, 0.0074, 0.0087, 1.4100, 1.9266, 0; % Channel 3 A02_normal_mat
    0.0899, 0.0175, 0.0075, 0.0084, 1.3260, 1.8895, 0; % Channel 4 A02_normal_mat

    0.0896,	0.0175,	0.0073,	0.0083,	1.3042,	1.8802,	0;
    0.0934,	0.0180,	0.0077,	0.0087,	1.3515,	1.9132,	0;
    0.0877,	0.0171,	0.0073,	0.0082,	1.2849,	1.8630,	0;
    0.0889, 0.0173	0.0073,	0.0083,	1.2905,	1.8694,	0;
];

X = feature_matrix(:, 1:end-1); % Features
Y = feature_matrix(:, end); % Labels

% Split data into training and testing sets (20% training, 80% testing)
rng(1); % For reproducibility
cv = cvpartition(size(X, 1), 'HoldOut', 0.1); % Change HoldOut to 0.8 for 80% testing
X_train = X(training(cv), :);
Y_train = Y(training(cv), :);
X_test = X(test(cv), :);
Y_test = Y(test(cv), :);



% Create and train the K-NN model
K = 16; % Number of neighbors
mdl = fitcknn(X_train, Y_train, 'NumNeighbors', K);

% TRAINING MODEL ENDS_____________________________________________________________________________


%%
%%
%%
%%
%%
%%
%%


%INPUT WHATEVER YOU WANT INTO THE HELP VARIABLE - YOU CAN EVEN IMPORT FILE



%Predict labels for the test set
%Y_pred = predict(mdl, X_test);
%display(Y_pred)

help = [0.0896,	0.0175,	0.0073,	0.0083,	1.3042,	1.8802;
        0.0934,	0.0180,	0.0077,	0.0087,	1.3515,	1.9132;
        0.0877,	0.0171,	0.0073,	0.0082,	1.2849,	1.8630;
        0.0889, 0.0173	0.0073,	0.0083,	1.2905,	1.8694; ]; %Put MATRIX IN HERE CHANGE VARIABLE HOWEVER YOU WOULD LIKE TO

pred = mode(predict(mdl, help));


if pred == 0
    final_pred_dis = 'Stress';
else
    final_pred_dis = "No Stress";
end

display(final_pred_dis)
% Separation between the training part
% Result as Word stress or no stress

% Evaluate the model
accuracy = sum(Y_pred == Y_test) / numel(Y_test);
fprintf('Accuracy of the K-NN model: %.2f%%\n', accuracy * 100);
