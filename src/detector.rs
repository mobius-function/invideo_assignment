use anyhow::{Context, Result};
use image::DynamicImage;
use rustface::{Detector, ImageData};
use std::path::Path;

/// Represents a detected face with bounding box and confidence
#[derive(Debug, Clone)]
pub struct FaceBox {
    pub x: i32,      // Left coordinate
    pub y: i32,      // Top coordinate
    pub width: i32,  // Width of bounding box
    pub height: i32, // Height of bounding box
    pub confidence: f32, // Detection confidence (0.0-1.0)
}

/// Trait for face detector implementations
pub trait FaceDetector {
    /// Initialize a new detector
    fn new() -> Result<Self> where Self: Sized;

    /// Detect faces in an image
    fn detect_faces(&mut self, image: &DynamicImage, threshold: f32) -> Result<Vec<FaceBox>>;

    /// Optional method to set detector-specific parameters
    fn set_params(&mut self, _params: &str) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }
}

/// RustFace (SeetaFace) detector implementation
pub struct RustFaceDetector {
    detector: Box<dyn Detector>,
}

impl FaceDetector for RustFaceDetector {
    fn new() -> Result<Self> {
        // Download the model file if it doesn't exist
        let model_path = "model/seeta_fd_frontal_v1.0.bin";

        if !Path::new(model_path).exists() {
            println!("Downloading face detection model...");

            // Create the model directory
            std::fs::create_dir_all("model")?;

            // Try multiple URLs for the model
            let model_urls = [
                // Direct link from the raw GitHub content
                "https://github.com/atomashpolskiy/rustface/raw/master/model/seeta_fd_frontal_v1.0.bin",
                // Alternative raw content URL
                "https://raw.githubusercontent.com/atomashpolskiy/rustface/master/model/seeta_fd_frontal_v1.0.bin",
            ];

            let mut downloaded = false;
            let mut last_error = None;

            for url in &model_urls {
                println!("Trying to download from: {}", url);

                match ureq::get(url).call() {
                    Ok(response) => {
                        let mut reader = response.into_reader();
                        let mut file = std::fs::File::create(model_path)?;
                        std::io::copy(&mut reader, &mut file)?;
                        println!("Model downloaded successfully from {}", url);
                        downloaded = true;
                        break;
                    }
                    Err(err) => {
                        println!("Failed to download from {}: {}", url, err);
                        last_error = Some(err);
                        continue;
                    }
                }
            }

            if !downloaded {
                return Err(anyhow::anyhow!(
                    "Failed to download model from all sources. Last error: {:?}\n\
                    Please download the model manually from:\n\
                    https://github.com/atomashpolskiy/rustface/tree/master/model\n\
                    and place it at: {}", 
                    last_error,
                    model_path
                ));
            }
        } else {
            println!("Model already exists at: {}", model_path);
        }

        // Create the detector
        let detector = rustface::create_detector(model_path)
            .context("Failed to create face detector")?;

        Ok(Self { detector })
    }

    fn detect_faces(&mut self, image: &DynamicImage, threshold: f32) -> Result<Vec<FaceBox>> {
        let gray_image = image.to_luma8();

        // Convert to rustface ImageData format
        let (width, height) = gray_image.dimensions();
        let mut image_data = ImageData::new(gray_image.as_raw(), width, height);

        // Detect faces
        let faces = self.detector.detect(&mut image_data);

        // Convert to our FaceBox format, filtering by threshold
        let mut result = Vec::new();
        for face in faces {
            if face.score() >= f64::from(threshold) {
                let bbox = face.bbox();
                result.push(FaceBox {
                    x: bbox.x() as i32,
                    y: bbox.y() as i32,
                    width: bbox.width() as i32,
                    height: bbox.height() as i32,
                    confidence: face.score() as f32,
                });
            }
        }

        Ok(result)
    }
}

// Factory function to create detectors by name
pub fn create_detector(name: &str) -> Result<Box<dyn FaceDetector>> {
    match name.to_lowercase().as_str() {
        "rustface" => Ok(Box::new(RustFaceDetector::new()?)),
        // Add other detectors here as needed
        _ => Err(anyhow::anyhow!("Unknown detector: {}", name)),
    }
}

