The implementation handles key edge cases as follows: 
1. Multiple faces are detected independently without interference using confidence-based selection
2. Boundary faces are adjusted to remain within valid image coordinates
3. Consistent square crops maintain dataset uniformity regardless of original face shape
4. Face quality is indicated through confidence scores in filenames

Why We Used SeetaFace:
We chose SeetaFace (via RustFace library) because: 
1. it provides a pure Rust implementation without complex dependencies
2. works efficiently on CPU-only systems without requiring GPU acceleration
3. offers good performance for face detection across various poses and conditions
4. Its 15.7MB model size balances accuracy and resource efficiency.

How to run?
# Download the WIDER FACE dataset
cargo run --bin download_wider_face

# Run face detection and cropping
cargo run --release -- --input-dir=data/input/wider_face --output-dir=data/output

# With custom settings
cargo run --release -- --input-dir=data/input/wider_face --output-dir=data/output --threshold=0.4 --size=256 --max-faces=8000

# To see all options
cargo run --release -- --help
