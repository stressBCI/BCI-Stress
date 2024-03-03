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
