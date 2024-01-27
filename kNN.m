% Assuming feature_matrix is already computed as shown in the previous code

% Initialize the new column for stress classification
stress_labels = zeros(size(feature_matrix, 1), 1);

% Thresholds for stress classification (we need to change these into the actual analytical values)
threshold_RER_delta = 0.5; #
threshold_RER_theta = 0.5;
threshold_RER_alpha = 0.5;
threshold_RER_beta = 0.5;
threshold_SC = 100; % Adjust as needed
threshold_SE = 2;   % Adjust as needed

% Analytical skills for stress classification
for i = 1:size(feature_matrix, 1)
    % Apply your analytical skills here to determine stress or not
    if feature_matrix(2, i) > feature_matirx(
       feature_matrix(4, i) > threshold_RER_beta 
       feature_matrix(5, i) > threshold_SC 
       feature_matrix(6, i) > threshold_SE
        stress_labels(i) = 1; % Indicates stress
    else
        stress_labels(i) = 0; % Indicates no stress
    end
end

% Add the new column to the existing feature matrix
feature_matrix_with_labels = [feature_matrix, stress_labels];

% Display the updated feature matrix
disp(feature_matrix_with_labels);

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
