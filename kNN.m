% Sample feature matrix
feature_matrix = [
    0.25, 0.30, 0.15, 0.30, 10.5, 0.85, 1; % Channel 1
    0.20, 0.25, 0.20, 0.35, 11.2, 0.78, 0; % Channel 2
    0.30, 0.35, 0.10, 0.25, 9.8, 0.92, 1; % Channel 3
    0.15, 0.40, 0.25, 0.20, 10.9, 0.75, 0; % Channel 4
    0.18, 0.28, 0.22, 0.32, 11.0, 0.80, 1; % Channel 1
    0.22, 0.32, 0.18, 0.28, 10.3, 0.88, 0; % Channel 2
    0.28, 0.20, 0.30, 0.22, 10.7, 0.83, 1; % Channel 3
    0.35, 0.15, 0.25, 0.25, 11.5, 0.70, 0; % Channel 4
];

% Assuming the last column of feature_matrix contains the labels (stress or no stress)
X = feature_matrix(:, 1:end-1); % Features
Y = feature_matrix(:, end); % Labels

% Split data into training and testing sets (80% training, 20% testing)
rng(1); % For reproducibility
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
Y_train = Y(training(cv), :);
X_test = X(test(cv), :);
Y_test = Y(test(cv), :);

% Create and train the K-NN model
K = 5; % Number of neighbors
mdl = fitcknn(X_train, Y_train, 'NumNeighbors', K);

% Predict labels for the test set
Y_pred = predict(mdl, X_test);

% Evaluate the model
accuracy = sum(Y_pred == Y_test) / numel(Y_test);
fprintf('Accuracy of the K-NN model: %.2f%%\n', accuracy * 100);
