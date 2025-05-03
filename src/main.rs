mod detector;

use anyhow::{Context, Result};
use clap::Parser;
use detector::{create_detector, FaceDetector};
use image::GenericImageView;
use log::{debug, error, info, warn};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

/// Command line arguments
#[derive(Parser, Debug)]
#[clap(author, version, about = "Extract and crop faces from images using face detection")]
struct Args {
    /// Input directory containing images
    #[clap(short, long, value_parser)]
    input_dir: PathBuf,

    /// Output directory for cropped faces
    #[clap(short, long, value_parser)]
    output_dir: PathBuf,

    /// Confidence threshold for face detection (0.0-1.0)
    #[clap(short, long, default_value = "0.5")]
    threshold: f32,

    /// Maximum number of faces to extract (0 for unlimited)
    #[clap(short, long, default_value = "10000")]
    max_faces: usize,

    /// Batch size for processing
    #[clap(short, long, default_value = "16")]
    batch_size: usize,

    /// Square size for output faces (px)
    #[clap(short, long, default_value = "128")]
    size: u32,

    /// Face detector to use (rustface, etc.)
    #[clap(long, default_value = "rustface")]
    detector: String,

    /// Optional detector-specific parameters (JSON string)
    #[clap(long, default_value = "")]
    detector_params: String,
}

/// Process an image file and save cropped faces
fn process_image(
    path: &Path,
    detector: &mut Box<dyn FaceDetector>, // Changed to &mut
    output_dir: &Path,
    threshold: f32,
    size: u32,
    face_counter: &mut usize
) -> Result<usize> {
    // Load image
    let img = image::open(path)
        .with_context(|| format!("Failed to open image: {:?}", path))?;

    // Detect faces
    let faces = detector.detect_faces(&img, threshold)?;

    // Process each detected face
    let mut faces_found = 0;

    for face in faces {
        // Crop face with some padding
        let padding_factor = 0.5; // 50% extra padding around face
        let padding_w = (face.width as f32 * padding_factor) as i32;
        let padding_h = (face.height as f32 * padding_factor) as i32;

        let x = (face.x - padding_w / 2).max(0);
        let y = (face.y - padding_h / 2).max(0);
        let width = (face.width + padding_w).min(img.width() as i32 - x);
        let height = (face.height + padding_h).min(img.height() as i32 - y);

        // Ensure we have a valid crop region
        if width <= 0 || height <= 0 {
            continue;
        }

        // Get square crop (use the smaller dimension)
        let size_to_use = width.min(height);
        let x_center = x + width / 2;
        let y_center = y + height / 2;
        let x_crop = (x_center - size_to_use / 2).max(0);
        let y_crop = (y_center - size_to_use / 2).max(0);

        // Create the crop
        let cropped = img.crop_imm(
            x_crop as u32,
            y_crop as u32,
            size_to_use as u32,
            size_to_use as u32
        );

        // Resize to the requested size
        let resized = cropped.resize_exact(
            size,
            size,
            image::imageops::FilterType::Lanczos3
        );

        // Generate output filename with face index and confidence
        let filename = format!(
            "face_{:06}_{:.3}.jpg",
            face_counter,
            face.confidence
        );
        let output_path = output_dir.join(filename);

        // Save the cropped and resized face
        resized.save(&output_path)
            .with_context(|| format!("Failed to save cropped face to: {:?}", output_path))?;

        debug!("Saved face from {:?} to {:?}", path, output_path);

        *face_counter += 1;
        faces_found += 1;
    }

    Ok(faces_found)
}

/// Main program logic
fn run(args: Args) -> Result<()> {
    // Initialize logger
    env_logger::init();

    // Create output directory if it doesn't exist
    fs::create_dir_all(&args.output_dir)
        .context("Failed to create output directory")?;

    // Initialize face detector
    info!("Initializing face detector: {}", args.detector);
    let mut detector = create_detector(&args.detector)
        .context("Failed to initialize face detector")?;

    // Set detector params if provided
    if !args.detector_params.is_empty() {
        detector.set_params(&args.detector_params)?;
    }

    // Find all image files in input directory
    info!("Scanning input directory for images: {:?}", args.input_dir);
    let image_paths: Vec<PathBuf> = WalkDir::new(&args.input_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            if let Some(ext) = e.path().extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                return ["jpg", "jpeg", "png", "bmp"].contains(&ext_str.as_str());
            }
            false
        })
        .map(|e| e.path().to_owned())
        .collect();

    info!("Found {} images", image_paths.len());

    if image_paths.is_empty() {
        warn!("No images found in input directory");
        return Ok(());
    }

    // Process images in chunks
    let mut face_counter = 0;
    let mut processed_counter = 0;
    let start_time = Instant::now();

    for (batch_idx, chunk) in image_paths.chunks(args.batch_size).enumerate() {
        // Check if we've reached the maximum number of faces
        if args.max_faces > 0 && face_counter >= args.max_faces {
            info!("Reached maximum number of faces ({}), stopping", args.max_faces);
            break;
        }

        info!(
            "Processing batch {}/{} ({} images)",
            batch_idx + 1,
            (image_paths.len() + args.batch_size - 1) / args.batch_size,
            chunk.len()
        );

        // Process each image in the batch
        for path in chunk {
            match process_image(path, &mut detector, &args.output_dir, args.threshold, args.size, &mut face_counter) { // Changed to &mut detector
                Ok(_faces_found) => {
                    processed_counter += 1;
                    if processed_counter % 10 == 0 {
                        let elapsed = start_time.elapsed().as_secs();
                        if elapsed > 0 {
                            let images_per_sec = processed_counter as f64 / elapsed as f64;
                            info!(
                                "Processed {}/{} images ({:.2} images/sec), found {} faces",
                                processed_counter,
                                image_paths.len(),
                                images_per_sec,
                                face_counter
                            );
                        }
                    }
                },
                Err(err) => {
                    error!("Failed to process {:?}: {}", path, err);
                    processed_counter += 1;
                }
            }
        }

        info!(
            "Processed {} faces so far",
            face_counter
        );
    }

    let elapsed = start_time.elapsed().as_secs();

    info!(
        "Finished processing. Extracted {} faces in {} seconds",
        face_counter,
        elapsed
    );

    if face_counter < 4000 {
        warn!(
            "Only extracted {} faces, which is less than the recommended minimum of 4,000",
            face_counter
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}
