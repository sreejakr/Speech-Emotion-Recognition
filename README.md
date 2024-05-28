# Speech-Emotion-Recognition

About Dataset
Context
Speech Emotion Recognition (SER) is a process of identifying human emotions from speech signals. It combines elements from signal processing, machine learning, and cognitive science to analyze vocal expressions and classify them into different emotional categories. 

Content
Here 4 most popular datasets in English: Crema, Ravdess, Savee and Tess. Each of them contains audio in .wav format with some main labels.

Ravdess:

Here is the filename identifiers as per the official RAVDESS website:

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
So, here's an example of an audio filename. 02-01-06-01-02-01-12.wav This means the meta data for the audio file is:

Video-only (02)
Speech (01)
Fearful (06)
Normal intensity (01)
Statement "dogs" (02)
1st Repetition (01)
12th Actor (12) - Female (as the actor ID number is even)
Crema:

The third component is responsible for the emotion label:

SAD - sadness;
ANG - angry;
DIS - disgust;
FEA - fear;
HAP - happy;
NEU - neutral.
Tess:

Very similar to Crema - label of emotion is contained in the name of file.

Savee:

The audio files in this dataset are named in such a way that the prefix letters describes the emotion classes as follows:

'a' = 'anger'
'd' = 'disgust'
'f' = 'fear'
'h' = 'happiness'
'n' = 'neutral'
'sa' = 'sadness'
'su' = 'surprise'

dataset link - https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en/data

# Analysis

The figures below represent the distribution of the labels or emotions in the dataset. 

<img width="789" alt="Screenshot 2024-05-28 at 16 01 45" src="https://github.com/sreejakr/Speech-Emotion-Recognition/assets/58878572/ff6c7299-1ba5-48d7-a425-41b6dcdb4efd">

## Waveforms

Waveforms are graphical representations of how a signal varies with time. They are fundamental in various fields such as physics, engineering, audio processing, and neuroscience. Here’s a detailed explanation of what waveforms are and their significance:

Definition and Basic Concepts
Waveform Definition:

A waveform depicts the variation of a quantity (such as voltage, current, sound pressure, or displacement) as a function of time.
It is typically represented on a two-dimensional graph where the x-axis represents time, and the y-axis represents the amplitude of the signal.
Amplitude:

The amplitude of a waveform is the height of the wave, indicating the strength or intensity of the signal at any given point in time.
Frequency:

Frequency is the number of times the waveform repeats itself within a specific time period (usually one second), measured in Hertz (Hz).
Period:

The period is the duration of one complete cycle of the waveform, which is the inverse of the frequency.
Phase:

Phase refers to the position of a point in time on the waveform cycle, usually measured in degrees or radians.
Wavelength:

Wavelength is the distance between successive corresponding points of the waveform (such as peak to peak or trough to trough) in space.


## Mel Spectrogram
**Definition**
A mel spectrogram is a type of spectrogram that represents the power spectral density of a sound signal on the mel scale, which is a perceptual scale of pitches judged by listeners to be equal in distance from one another.

**Characteristics**
Time-Frequency Representation: Shows how the spectral content of a signal varies over time.
Mel Scale: The frequency axis is converted to the mel scale to better align with human perception of pitch.
Intensity: The color or brightness in the spectrogram indicates the intensity of the frequencies at a given time.
Calculation
Short-Time Fourier Transform (STFT): Computes the Fourier transform of short overlapping windows of the audio signal to obtain the spectrogram.
Mel Filter Bank: Applies a filter bank that maps the frequency bins of the spectrogram to the mel scale.
Logarithm: Often, the logarithm of the mel spectrogram values is taken to compress the dynamic range.

## Mel-Frequency Cepstral Coefficients (MFCCs)
**Definition**
MFCCs are coefficients that collectively make up an MFC (mel-frequency cepstrum), which represents the short-term power spectrum of a sound signal on the mel scale.

**Characteristics**
Compact Representation: Provides a compact and comprehensive representation of the spectral properties of a signal.
Cepstral Domain: Derived from the inverse Fourier transform of the logarithm of the estimated signal spectrum.
Static and Dynamic Features: Can include both static coefficients and their first and second derivatives (delta and delta-delta coefficients).

<img width="429" alt="Screenshot 2024-05-28 at 17 02 13" src="https://github.com/sreejakr/Speech-Emotion-Recognition/assets/58878572/8beb807b-7c47-47ce-9828-575d19ab9921">

<img width="427" alt="Screenshot 2024-05-28 at 17 02 04" src="https://github.com/sreejakr/Speech-Emotion-Recognition/assets/58878572/ed674e42-7802-4b58-88be-45e4a5fe58e2">

<img width="445" alt="Screenshot 2024-05-28 at 17 01 54" src="https://github.com/sreejakr/Speech-Emotion-Recognition/assets/58878572/068144b4-08b0-4d8f-b90d-a27bedf84cd0">

<img width="424" alt="Screenshot 2024-05-28 at 17 01 40" src="https://github.com/sreejakr/Speech-Emotion-Recognition/assets/58878572/da98b1ec-a7f8-4b78-95fb-7211ee29af2f">

<img width="435" alt="Screenshot 2024-05-28 at 17 01 30" src="https://github.com/sreejakr/Speech-Emotion-Recognition/assets/58878572/d1bd8161-5916-4c6b-83e3-abbe5faa50e0">

<img width="424" alt="Screenshot 2024-05-28 at 17 01 21" src="https://github.com/sreejakr/Speech-Emotion-Recognition/assets/58878572/a2c36fa9-d0de-4e66-b242-c3e9f6a77400">

<img width="430" alt="Screenshot 2024-05-28 at 17 01 09" src="https://github.com/sreejakr/Speech-Emotion-Recognition/assets/58878572/0652cb6d-4178-4d19-8d00-1b4f3e65c00c">



-----------------------------------------------------------------------------------------

# Audio Augmentation

# Audio Augmentation

Explanation of Audio Augmentation Techniques
Audio augmentation techniques are used to artificially increase the size and diversity of a dataset by applying various transformations to the audio signals. These techniques help improve the robustness and generalization of machine learning models. Below are the explanations for each of the provided augmentation functions:

1. Adding White Noise

White Noise: This function adds white noise to the original signal. White noise is a random signal having equal intensity at different frequencies, giving it a constant power spectral density.
Implementation:

Purpose: Adding white noise can make the model more robust to variations in the data and helps prevent overfitting by simulating a real-world scenario where recordings might have background noise.


2. Time Stretching

Time Stretching: This technique changes the speed of the audio without affecting its pitch.

Purpose: Time stretching helps the model learn to recognize features regardless of the speed of speech or music. It simulates variations in speaking or playing speed.


3. Pitch Scaling

Pitch Shifting: This technique changes the pitch of the audio without affecting its duration.

Purpose: Pitch scaling allows the model to learn to recognize features regardless of variations in pitch, which is useful in music and speech applications where pitch can vary naturally.


4. Random Gain

Random Gain: This technique randomly changes the volume of the audio signal.

Implementation:
Generate a random gain rate using random.uniform within the specified range (min_factor to max_factor).
Multiply the original signal by this gain rate.

Purpose: Applying random gain helps the model become invariant to volume changes, making it more robust to different recording conditions and speaker volumes.


5. Invert Polarity

Polarity Inversion: This technique flips the audio signal upside down.

Implementation:
Simply multiply the signal by -1.

Purpose: Inverting polarity is a simple yet effective way to augment audio data. It does not affect the perceived sound but changes the waveform, helping the model become robust to polarity inversion which might occur in some recording setups.


These audio augmentation techniques are valuable for enhancing the robustness and generalization capabilities of machine learning models by artificially increasing the diversity of the training data. They simulate various real-world variations and distortions, making the models more adaptable and effective in different scenarios.


Original Audio

<img width="763" alt="Screenshot 2024-05-28 at 17 14 27" src="https://github.com/sreejakr/Speech-Emotion-Recognition/assets/58878572/566c75dd-dcc1-4c67-b06d-1f2f9447e393">

Augmented signal
<img width="771" alt="Screenshot 2024-05-28 at 17 15 40" src="https://github.com/sreejakr/Speech-Emotion-Recognition/assets/58878572/0fe68677-78c3-4db3-bf2d-0015cabadc9b">

<img width="764" alt="Screenshot 2024-05-28 at 17 15 26" src="https://github.com/sreejakr/Speech-Emotion-Recognition/assets/58878572/b3369a7a-6e77-42f1-815a-469a4561d506">

<img width="769" alt="Screenshot 2024-05-28 at 17 15 46" src="https://github.com/sreejakr/Speech-Emotion-Recognition/assets/58878572/4feac5fe-67f9-403b-a587-1b23d1f3cf38">

<img width="764" alt="Screenshot 2024-05-28 at 17 15 34" src="https://github.com/sreejakr/Speech-Emotion-Recognition/assets/58878572/3843fc19-4667-4e40-971a-d23ba4a73cdb">





