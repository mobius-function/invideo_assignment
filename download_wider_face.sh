use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const VALIDATION_SET_URL: &str = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip";
const ANNOTATION_URL: &str = "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip";

fn main() -> io::Result<()> {
    // Create directory structure
    println!("Creating directories...");
    create_directories()?;
    
    // Download the dataset
    println!("Downloading WIDER FACE dataset from Hugging Face...");
    download_dataset()?;
    
    // Extract dataset
    extract_files()?;
    
    // Set up input/output directories
    setup_input_dir()?;
    
    println!("Setup complete! You can now access the images from data/input/wider_face");
    Ok(())
}

fn create_directories() -> io::Result<()> {
    fs::create_dir_all("data/wider_face")?;
    fs::create_dir_all("data/input")?;
    fs::create_dir_all("data/output")?;
    Ok(())
}

fn download_dataset() -> io::Result<()> {
    println!("Downloading validation set...");
    download_file(VALIDATION_SET_URL, "data/wider_face/WIDER_val.zip")?;
    
    println!("Downloading annotations...");
    download_file(ANNOTATION_URL, "data/wider_face/wider_face_split.zip")?;
    
    println!("Download complete!");
    Ok(())
}

fn download_file(url: &str, dest: &str) -> io::Result<()> {
    // Check if file already exists
    if Path::new(dest).exists() {
        println!("File {} already exists, skipping download.", dest);
        return Ok(());
    }
    
    // Create a temporary file for download
    let temp_dest = format!("{}.tmp", dest);
    let mut file = File::create(&temp_dest)?;
    
    // Download the file using ureq
    println!("Downloading from {}...", url);
    
    match ureq::get(url).call() {
        Ok(response) => {
            let mut reader = response.into_reader();
            let bytes_copied = io::copy(&mut reader, &mut file)?;
            println!("Downloaded {} bytes", bytes_copied);
        },
        Err(err) => {
            // Remove temporary file
            drop(file);
            fs::remove_file(&temp_dest)?;
            
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to download: {}", err)
            ));
        }
    }
    
    // Rename the temporary file to the final destination
    fs::rename(temp_dest, dest)?;
    println!("Downloaded file to {}", dest);
    
    Ok(())
}

fn extract_files() -> io::Result<()> {
    println!("Extracting validation set...");
    extract_zip("data/wider_face/WIDER_val.zip", "data/wider_face")?;
    
    println!("Extracting annotations...");
    extract_zip("data/wider_face/wider_face_split.zip", "data/wider_face")?;
    
    println!("Extraction complete!");
    Ok(())
}

fn extract_zip(zip_path: &str, extract_to: &str) -> io::Result<()> {
    let file = File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = match file.enclosed_name() {
            Some(path) => Path::new(extract_to).join(path),
            None => continue,
        };
        
        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p)?;
                }
            }
            let mut outfile = File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }
    
    Ok(())
}

fn setup_input_dir() -> io::Result<()> {
    let source_dir = Path::new("data/wider_face/WIDER_val/images");
    let target_dir = Path::new("data/input/wider_face");
    
    if !source_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Source directory not found: {:?}", source_dir)
        ));
    }
    
    // Remove target if it exists
    if target_dir.exists() {
        if target_dir.is_dir() {
            fs::remove_dir_all(target_dir)?;
        } else {
            fs::remove_file(target_dir)?;
        }
    }
    
    // Create symbolic link or copy
    #[cfg(target_family = "unix")]
    {
        println!("Creating symbolic link...");
        std::os::unix::fs::symlink(source_dir, target_dir)?;
    }
    
    #[cfg(target_family = "windows")]
    {
        // Try symbolic link first (needs admin rights)
        println!("Trying to create symbolic link...");
        let result = Command::new("cmd")
            .args(&["/C", "mklink", "/D", 
                   &target_dir.to_string_lossy().replace("/", "\\"), 
                   &source_dir.to_string_lossy().replace("/", "\\")])
            .status();
            
        // If symlink fails, copy the directory
        if result.is_err() || !result.unwrap().success() {
            println!("Creating symbolic link failed, copying files instead...");
            copy_dir_all(source_dir, target_dir)?;
        }
    }
    
    println!("Input directory set up at {:?}", target_dir);
    Ok(())
}

#[cfg(target_family = "windows")]
fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(&dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());
        
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &dst_path)?;
        } else {
            fs::copy(entry.path(), dst_path)?;
        }
    }
    Ok(())
}