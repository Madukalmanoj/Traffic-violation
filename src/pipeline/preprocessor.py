import cv2
import numpy as np

class SmartQualityAnalyzer:
    """
    Analyzes an image and calculates a quality score from 0 to 100 based on:
    - Sharpness (Laplacian variance)
    - Contrast (Standard deviation of luminance)
    - Brightness (Mean luminance compared to ideal 128)
    """
    @staticmethod
    def analyze(img):
        # Convert to grayscale for sharpness analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness calculation (Laplacian variance)
        sharpness_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Scale to 0-100: variance of ~500+ is considered sharp
        sharpness_score = min(100.0, sharpness_var / 5.0)
        
        # Convert to LAB for contrast and brightness analysis
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)
        
        # 2. Contrast calculation (Std Dev of L channel)
        contrast_std = np.std(l_channel)
        # Scale to 0-100: std dev of ~50 is high contrast
        contrast_score = min(100.0, contrast_std * 2.0)
        
        # 3. Brightness calculation (Mean of L channel compared to ideal 128)
        brightness_mean = np.mean(l_channel)
        # Absolute distance from 128 (ideal midtone brightness)
        brightness_score = max(0.0, 100.0 - abs(brightness_mean - 128.0) * 1.2)
        
        # Weighted overall score (40% sharpness, 30% contrast, 30% brightness)
        overall_score = 0.4 * sharpness_score + 0.3 * contrast_score + 0.3 * brightness_score
        
        metrics = {
            "sharpness": float(sharpness_var),
            "sharpness_score": float(sharpness_score),
            "contrast": float(contrast_std),
            "contrast_score": float(contrast_score),
            "brightness": float(brightness_mean),
            "brightness_score": float(brightness_score),
            "overall_score": float(overall_score)
        }
        return metrics

class DCPDehazer:
    """
    Implements single-image dehazing based on the Dark Channel Prior (DCP) algorithm.
    """
    def __init__(self, win_size=15, omega=0.75, t0=0.1):
        self.win_size = win_size
        self.omega = omega
        self.t0 = t0

    def get_dark_channel(self, img):
        # Min across RGB channels
        min_channel = np.min(img, axis=2)
        # Morphological erosion is a minimum filter in a local neighborhood
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.win_size, self.win_size))
        dark = cv2.erode(min_channel, kernel)
        return dark

    def estimate_atmospheric_light(self, img, dark):
        h, w = dark.shape
        num_pixels = h * w
        top_k = max(1, int(num_pixels * 0.001)) # top 0.1% brightest pixels
        
        # Find indices of the top_k brightest pixels in the dark channel
        flat_dark = dark.flatten()
        indices = np.argsort(flat_dark)[-top_k:]
        
        # From these coordinates, select the pixel in original image with highest intensity
        flat_img = img.reshape(-1, 3)
        brightest_idx = indices[0]
        max_intensity = -1
        
        for idx in indices:
            intensity = np.mean(flat_img[idx])
            if intensity > max_intensity:
                max_intensity = intensity
                brightest_idx = idx
                
        # Atmospheric light A is the BGR color of this pixel
        A = flat_img[brightest_idx]
        return A

    def estimate_transmission(self, img, A):
        # Normalize image by atmospheric light
        # Handle division by zero
        A_safe = np.maximum(A, 1.0)
        norm_img = img / A_safe
        dark_norm = self.get_dark_channel(norm_img)
        transmission = 1.0 - self.omega * dark_norm
        return np.clip(transmission, self.t0, 1.0)

    def recover_scene(self, img, A, transmission):
        t_broadcast = np.expand_dims(np.maximum(transmission, self.t0), axis=2)
        # J = (I - A) / t + A
        recovered = (img - A) / t_broadcast + A
        return np.clip(recovered, 0.0, 255.0).astype(np.uint8)

    def dehaze(self, img):
        img_float = img.astype(np.float32)
        dark = self.get_dark_channel(img_float)
        A = self.estimate_atmospheric_light(img_float, dark)
        transmission = self.estimate_transmission(img_float, A)
        recovered = self.recover_scene(img_float, A, transmission)
        return recovered

class ImagePreprocessor:
    """
    Coordinates the 5-stage preprocessing pipeline:
    1. Smart Quality Analyzer (Laplacian/Contrast/Brightness)
    2. Adaptive Contrast Enhancement (CLAHE) - Applied to LAB L channel
    3. Single-Image Dehazing (Dark Channel Prior scene recovery)
    4. Edge-Preserving Denoising (Bilateral Filtering)
    5. Image Sharpening (Unsharp Masking)
    """
    def __init__(self, threshold=70.0):
        self.threshold = threshold
        self.dehazer = DCPDehazer()

    def enhance(self, img):
        if img is None:
            return None, {"overall_score": 100.0, "bypass": True}

        # Stage 1: Smart Quality Analyzer (Fast-path bypass)
        metrics = SmartQualityAnalyzer.analyze(img)
        score = metrics["overall_score"]
        
        if score > self.threshold:
            # Bypass heavy pipeline, run light sharpening pass only (Amount = 0.4)
            processed_img = self._apply_unsharp_mask(img, amount=0.4)
            metrics["bypass"] = True
            return processed_img, metrics
            
        # Score <= threshold: Run full active enhancement path
        metrics["bypass"] = False
        
        # Stage 2: Adaptive Contrast Enhancement (CLAHE in LAB space)
        img_clahe = self._apply_clahe(img)
        
        # Stage 3: Single-Image Dehazing (DCP)
        img_dehazed = self.dehazer.dehaze(img_clahe)
        
        # Stage 4: Edge-Preserving Denoising (Bilateral Filtering)
        # d=9, sigmaColor=75, sigmaSpace=75
        img_denoised = cv2.bilateralFilter(img_dehazed, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Stage 5: Image Sharpening (Unsharp Masking, Amount = 0.7)
        processed_img = self._apply_unsharp_mask(img_denoised, amount=0.7)
        
        return processed_img, metrics

    def _apply_clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def _apply_unsharp_mask(self, img, amount=0.4):
        # Sharpened = Original + Amount * (Original - Blurred)
        blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        return sharpened
