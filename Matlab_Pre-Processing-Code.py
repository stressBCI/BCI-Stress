% Load the EEG data
dataStruct = load('/Users/aarooshbalakrishnan/Documents/MATLAB/matlab.mat');

% Access the 'AOO1stress' field within the structure
EEG_data = dataStruct.AOO1stress;

% Display the first few rows of the EEG_data to inspect its structure
disp(EEG_data(1:5, :));% Sampling frequency
Fs = 250;

% Frequency bands for RER (in Hz)
delta_band = [1 3];
theta_band = [4 7];
alpha_band = [8 12];
beta_band = [13 30];

% Initialize the feature matrix
feature_matrix = [];

% Iterate over each channel to compute features
for channel = {'Fp1', 'Fp2', 'Fz', 'Cz'}
    data = EEG.(channel{1}); 
    
    % Compute the Power Spectral Density (PSD) using Welch's method
    [psd, freq] = pwelch(data, hamming(256), 128, 256, Fs);
    
    % Define the frequency indices for each band
    delta_idx = freq >= delta_band(1) & freq <= delta_band(2);
    theta_idx = freq >= theta_band(1) & freq <= theta_band(2);
    alpha_idx = freq >= alpha_band(1) & freq <= alpha_band(2);
    beta_idx = freq >= beta_band(1) & freq <= beta_band(2);
    
    % Calculate RER for each band
    RER_delta = sum(psd(delta_idx)) / sum(psd);
    RER_theta = sum(psd(theta_idx)) / sum(psd);
    RER_alpha = sum(psd(alpha_idx)) / sum(psd);
    RER_beta = sum(psd(beta_idx)) / sum(psd);
    
    % Calculate the Spectral Centroid
    SC = sum(freq .* psd) / sum(psd);
    
    % Calculate the Shannon Entropy
    psd_norm = psd / sum(psd); % Normalize the PSD to sum to 1
    SE = -sum(psd_norm .* log2(psd_norm + eps)); % eps to avoid log of 0
    
    % Append the features for this channel to the feature matrix
    features = [RER_delta, RER_theta, RER_alpha, RER_beta, SC, SE];
    feature_matrix = [feature_matrix; features];
end

% Transpose the feature matrix so that each column corresponds to a feature
% and each row corresponds to a channel
feature_matrix = feature_matrix.';

% Display the feature matrix
disp(feature_matrix);
