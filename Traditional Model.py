import numpy as np
import cv2
import pandas as pd
import openpyxl
from scipy import stats
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path

def analyze_surface_roughness(image_path):
    # Load and preprocess
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    
    # 1. Statistical Analysis
    std_dev = np.std(img)
    entropy = stats.entropy(np.histogram(img, bins=256)[0])
    
    # 2. Gradient Analysis
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    mean_gradient = np.mean(gradient_magnitude)
    
    # 3. GLCM Features
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # 4. FFT Analysis
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift))
    frequency_energy = np.sum(magnitude_spectrum)
    
    return {
        'statistical_roughness': std_dev,
        'entropy': entropy,
        'gradient_roughness': mean_gradient,
        'texture_contrast': contrast,
        'texture_homogeneity': homogeneity,
        'frequency_energy': frequency_energy
    }

def calibrate_measurements(known_samples):
    """
    Calibrate measurements using known roughness values
    known_samples: list of (image_path, actual_roughness) tuples
    """
    measurements = []
    actual_values = []
    
    for img_path, actual in known_samples:
        metrics = analyze_surface_roughness(img_path)
        measurements.append([metrics['statistical_roughness'], 
                           metrics['gradient_roughness'],
                           metrics['texture_contrast']])
        actual_values.append(actual)
    
    # Fit linear regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(measurements, actual_values)
    
    return model

def load_and_analyze_data(excel_path, image_folder):
    # Read Excel data
    df = pd.read_excel(excel_path)
    roughness_values = df.iloc[:, 1].values  # Second column
    
    # Process images
    image_folder = Path(image_folder)
    samples = []
    processed_results = []
    
    for i in range(1, 43):  # 1.jpg to 42.jpg
        img_path = image_folder / f"{i}.jpg"
        if img_path.exists():
            samples.append((str(img_path), roughness_values[i-1]))
            results = analyze_surface_roughness(str(img_path))
            processed_results.append({
                'image': i,
                'actual_roughness': roughness_values[i-1],
                **results
            })
    
    # Create calibration model
    model = calibrate_measurements(samples)
    
    # Save results
    results_df = pd.DataFrame(processed_results)
    results_df.to_excel('analysis_results.xlsx', index=False)
    
    return model, results_df


# Driver Code

model, results = load_and_analyze_data('roughness_data_Ra.xlsx', 'Training Images - Sorted')