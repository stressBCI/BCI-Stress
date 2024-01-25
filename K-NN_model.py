% Assuming 'feature_matrix' contains the features for training data
% 'labels' is a column vector containing corresponding labels (1 for stress, 0 for no stress)

% Example labels (replace this with actual labels)
labels = [1; 0; 1; 0]; % 1 for stress, 0 for no stress

% Assuming 'test_inputed_files' contains the feature matrix for test data add Test Data a file.

% Define the number of neighbors (k) for the K-NN model
k = 5;

% Train the K-NN model
mdl = fitcknn(feature_matrix, labels, 'NumNeighbors', k);

% Predict stress for the test data
predicted_labels = predict(mdl, test_inputed_files.');

% Display the predicted labels
disp('Predicted Labels:');
disp(predicted_labels);
