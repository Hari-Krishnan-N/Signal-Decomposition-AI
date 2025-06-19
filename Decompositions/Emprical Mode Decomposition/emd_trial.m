% Step 1: Create a signal composed of 10 modes (sinusoids with different frequencies)
Fs = 1000;                    % Sampling frequency (Hz)
t = 0:1/Fs:1;                 % Time vector (1 second)
frequencies = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]; % Frequencies for the 10 modes
amplitudes = 1 + 0.5 * rand(1, 10); % Random amplitudes for each mode

% Create the combined signal
signal = zeros(size(t));
for i = 1:10
    signal = signal + amplitudes(i) * sin(2 * pi * frequencies(i) * t);
end

% Step 2: Perform Empirical Mode Decomposition (EMD)
imfs = emd(signal,"SiftRelativeTolerance",0.00000001, "SiftMaxIterations",500); % Perform EMD

% Step 3: Plot the original signal and the IMFs
figure;

% Plot Original Signal and its Frequency Spectrum
subplot(2,2,1);
plot(t, signal);
title('Original Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Frequency spectrum of the original signal
n = length(signal); % Number of samples
f = (0:n-1)*(Fs/n); % Frequency vector
Y = fft(signal); % Fourier Transform of the signal
subplot(2,2,2);
plot(f, abs(Y));
title('Frequency Spectrum of Original Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 60]); % Limit x-axis to show up to 60 Hz

% Plot IMFs and their Frequency Spectra
nIMFs = size(imfs, 2);
for i = 1:nIMFs
    % Plot each IMF in time domain
    subplot(nIMFs, 2, 2*i-1);
    plot(t, imfs(:,i));
    title(['IMF ' num2str(i)]);
    xlabel('Time (s)');
    ylabel('Amplitude');
    
    % Frequency spectrum of each IMF
    Y_imf = fft(imfs(:,i)); % Fourier Transform of the IMF
    subplot(nIMFs, 2, 2*i);
    plot(f, abs(Y_imf));
    title(['Frequency Spectrum of IMF ' num2str(i)]);
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    xlim([0 60]); % Limit x-axis to show up to 60 Hz
end
