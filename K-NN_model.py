% Load the features
load('/Users/aarooshbalakrishnan/Documents/MATLAB/shannon_entropy_features.mat');
load('/Users/aarooshbalakrishnan/Documents/MATLAB/spectral_centroid_features.mat');
load('/Users/aarooshbalakrishnan/Documents/MATLAB/rer_features.mat');

% Combine features into a single matrix
feature_matrix = [];
channel_names = {};

for ch = 1:length(shannon_entropy_features)
    channel_names{end + 1} = shannon_entropy_features(ch).channel_name;
    feature_matrix(end + 1, 1) = shannon_entropy_features(ch).shannon_entropy;
    feature_matrix(end, 2) = spectral_centroid_features(ch).spectral_centroid;
    
    for i = 1:length(rer_features(ch).rer)
        feature_matrix(end, 2 + i) = rer_features(ch).rer(i);
    end
end

% Labels indicating stress level (1: Stressed, 0: Not Stressed)
labels = [ones(1, length(shannon_entropy_features) / 2), zeros(1, length(shannon_entropy_features) / 2)];

% Normalize the feature matrix
normalized_features = zscore(feature_matrix);

% Split the data into training and testing sets
rng(42); % for reproducibility
indices = randperm(length(labels));
training_ratio = 0.8; % 80% training, 20% testing

training_set = normalized_features(indices(1:round(training_ratio * end)), :);
training_labels = labels(indices(1:round(training_ratio * end)));

testing_set = normalized_features(indices(round(training_ratio * end) + 1:end), :);
testing_labels = labels(indices(round(training_ratio * end) + 1:end));

% Train the kNN model
k = 5; % Number of neighbors
mdl = fitcknn(training_set, training_labels, 'NumNeighbors', k);

% Predict stress levels on the testing set
predictions = predict(mdl, testing_set);

% Evaluate the model
confusion_matrix = confusionmat(testing_labels, predictions);
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix, 'all');

disp('Confusion Matrix:');
disp(confusion_matrix);
disp(['Accuracy: ', num2str(accuracy * 100), '%']);
