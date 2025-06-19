p% Load audio file
filename = 'example.wav';  % Path to the .wav file
[y, Fs] = audioread(filename);

% Display properties
disp(['Sampling Frequency: ', num2str(Fs), ' Hz']);
disp(['Number of Samples: ', num2str(length(y))]);

% Plot the waveform
t = (0:length(y)-1)/Fs;  % Time vector
figure;
plot(t, y);
xlabel('Time (s)');
ylabel('Amplitude');
title('Audio Signal Waveform');

% Play the audio (optional)
sound(y, Fs);
