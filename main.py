import os
import time
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import easyocr
from datetime import datetime
import torch
import logging
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from huggingface_hub import hf_hub_download
import warnings
import cv2
from threading import Lock
import uuid

warnings.filterwarnings("ignore", category=UserWarning, module="paddle.utils.cpp_extension")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6'

# Configuration 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
UPLOAD_FOLDER = 'static/uploads'
CROPPED_FOLDER = 'static/cropped_images'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CROPPED_FOLDER'] = CROPPED_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_models():
    try:
        import paddle
        use_gpu = paddle.device.is_compiled_with_cuda()
        logger.info(f"Using GPU: {use_gpu}")
        
        model_path = hf_hub_download(
            repo_id="Moankhaled10/expiry-detection",
            filename="best.pt",
            cache_dir="model_weights"
        )
        yolo_model = YOLO(model_path)
        logger.info("Successfully initialized YOLO model")
        
        paddle_ocr = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            use_gpu=False,
            det_db_score_mode="fast",
            det_db_unclip_ratio=2.0,
            cpu_threads=6,
            det_model_dir="/root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer",
            rec_model_dir="/root/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer"
        )
        logger.info("Successfully initialized PaddleOCR")
        
        easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
        logger.info("Successfully initialized EasyOCR")
        
        logger.info("Running PaddleOCR warm-up inference")
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        paddle_ocr.ocr(dummy_image, cls=True)
        logger.info("PaddleOCR warm-up completed")
        
        return yolo_model, paddle_ocr, easy_reader
        
    except Exception as e:
        logger.critical(f"Model initialization failed: {e}", exc_info=True)
        raise

# Initialize models at startup
yolo_model, paddle_ocr, easy_reader = initialize_models()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename):
    ext = original_filename.rsplit('.', 1)[1].lower()
    unique_id = uuid.uuid4().hex
    return f"{unique_id}.{ext}"

def crop_image(image, bbox):
    """Crop image to the bounding box coordinates with padding."""
    x1, y1, x2, y2 = map(int, bbox)
    padding = 15
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]

def preprocess_image(image):
    
    if image is None:
        return None
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Resize and sharpen
    resized = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    return sharpened

def extract_text_from_region(image, output_dir, cls_name, detection_num):
    """Enhanced text extraction with better preprocessing and OCR handling"""
    if image is None or image.size == 0:
        return [], None

    try:
        # Enhanced preprocessing pipeline
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        processed = cv2.fastNlMeansDenoising(processed, h=10)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed = cv2.filter2D(processed, -1, kernel)
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.dilate(processed, kernel, iterations=1)
        
        crop_filename = os.path.join(output_dir, f"cropped_{cls_name}_{detection_num}_preprocessed.jpg")
        cv2.imwrite(crop_filename, processed)
        
        extracted_text = []
        
        # Try PaddleOCR with compatible parameters
        for attempt in range(2):
            try:
                result = paddle_ocr.ocr(
                    processed if attempt == 0 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    cls=True,
                    det_db_box_thresh=0.7 if attempt == 0 else 0.5,
                    rec_image_shape="3, 64, 320" if attempt == 0 else "3, 48, 320"
                )
                
                if result and result[0]:
                    for line in result[0]:
                        if line and len(line) >= 2:
                            text_info = line[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = str(text_info[0]).strip()
                                confidence = float(text_info[1])
                                if confidence > 0.5 and text:
                                    extracted_text.append((text, confidence))
                    if extracted_text:
                        break
            except Exception as e:
                logger.warning(f"PaddleOCR attempt {attempt} failed: {str(e)}")
                continue

        # Fallback to EasyOCR
        if not extracted_text:
            try:
                for preprocess_type in [0, 1, 2]:
                    if preprocess_type == 0:
                        easy_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    elif preprocess_type == 1:
                        easy_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                    else:
                        easy_image = cv2.bitwise_not(processed)
                        easy_image = cv2.cvtColor(easy_image, cv2.COLOR_GRAY2RGB)
                    
                    easy_results = easy_reader.readtext(
                        easy_image,
                        detail=1,
                        paragraph=False,
                        min_size=10,
                        slope_ths=0.3,
                        ycenter_ths=0.5,
                        height_ths=0.5,
                        width_ths=0.5,
                        decoder='beamsearch',
                        beamWidth=5
                    )
                    
                    for _, text, conf in easy_results:
                        text = str(text).strip()
                        if text and conf > 0.4:
                            extracted_text.append((text, conf))
                    
                    if extracted_text:
                        break
            except Exception as e:
                logger.error(f"EasyOCR failed: {str(e)}")

        # Post-process extracted text
        if extracted_text:
            merged_texts = {}
            for text, conf in extracted_text:
                text_lower = text.lower()
                if text_lower in merged_texts:
                    if conf > merged_texts[text_lower][1]:
                        merged_texts[text_lower] = (text, conf)
                else:
                    merged_texts[text_lower] = (text, conf)
            
            extracted_text = list(merged_texts.values())
            extracted_text.sort(key=lambda x: x[1], reverse=True)

        return extracted_text, crop_filename

    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return [], None

def process_image(image, output_dir, filename):
    
    try:
        # Same confidence threshold as standalone
        results = yolo_model.predict(source=image, conf=0.5)
    except Exception as e:
        logger.error(f"YOLO inference error: {e}")
        return {'error': f"YOLO inference error: {str(e)}"}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = 0
    detection_results = []
    cropped_images = []

    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    ax_main = fig.add_subplot(1, 2, 1)
    ax_main.imshow(image_rgb)
    ax_main.set_title(f"Detections for {filename}")
    ax_main.axis('off')

    try:
        for result in results:
            for box in result.boxes:
                cls_name = result.names[int(box.cls)]
                # Process only 'date' class
                if cls_name != 'date':
                    continue
                detections += 1
                conf = box.conf.item()
                bbox = box.xyxy[0].cpu().numpy()

                cropped_region = crop_image(image, bbox)
                if cropped_region is None:
                    continue

                try:
                    text_results, crop_filename = extract_text_from_region(
                        cropped_region, output_dir, cls_name, detections)
                except Exception as e:
                    logger.error(f"OCR extraction error: {e}")
                    text_results = []
                    crop_filename = None

                detection_results.append({
                    'class': cls_name,
                    'confidence': float(conf),
                    'bbox': bbox.tolist(),
                    'text': text_results,
                    'crop_path': crop_filename
                })

                # Add to visualization 
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
                ax_main.add_patch(rect)

                text_label = f"Date #{detections} \n"
                if text_results:
                    text_label += "\n".join([f"{text} (OCR: {text_conf:.2f})" for text, text_conf in text_results])
                else:
                    text_label += "No text detected"
                ax_main.text(x1, y1 - 10, text_label, bbox=dict(facecolor='white', alpha=0.8), fontsize=10, color='black')

                if cropped_region is not None and crop_filename:
                    cropped_rgb = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB)
                    cropped_rgb = cv2.resize(cropped_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    cropped_images.append({
                        'image': cropped_rgb,
                        'path': crop_filename,
                        'text': text_results
                    })

        if detections == 0:
            detection_results.append({'message': 'No date regions detected'})

        # Visualize cropped images
        if cropped_images:
            num_crops = len(cropped_images)
            for i, cropped_data in enumerate(cropped_images):
                ax_crop = fig.add_subplot(num_crops, 2, 2 * (i + 1))
                ax_crop.imshow(cropped_data['image'])
                title = f"Cropped Date #{i+1}\n"
                title += "\n".join([f"{text} (OCR: {conf:.2f})" for text, conf in cropped_data['text']]) if cropped_data['text'] else "No text"
                ax_crop.set_title(title)
                ax_crop.axis('off')

    except Exception as e:
        logger.error(f"Processing error: {e}")
        return {'error': f"Processing error: {str(e)}"}

    # Save visualization to base64
    try:
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        buf.seek(0)
        visualization = base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        visualization = None

    return {
        'detections': detection_results,
        'cropped_images': cropped_images,
        'visualization': visualization,
        'num_detections': detections
    }

# API Endpoints
@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API endpoint for image upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Generate unique filename to prevent collisions
        filename = generate_unique_filename(secure_filename(file.filename))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        result = process_image(image, app.config['CROPPED_FOLDER'], filename)
        result['original_image'] = f"/{filepath}"
        
        # Save processed image
        processed_filename = f"processed_{filename}"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        cv2.imwrite(processed_path, image)
        result['processed_image'] = f"/{processed_path}"

        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/capture', methods=['POST'])
def api_capture():
    """API endpoint for processing captured images"""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        image_data = data['image_data']
        if not image_data.startswith('data:image/'):
            return jsonify({'error': 'Invalid image data format'}), 400

        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        if len(image_bytes) > 12 * 1024 * 1024:
            return jsonify({'error': 'Image too large (max 12MB)'}), 400

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, image)

        result = process_image(image, app.config['CROPPED_FOLDER'], filename)
        result['original_image'] = f"/{filepath}"
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API capture error: {e}")
        return jsonify({'error': str(e)}), 500

# Web Routes
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)

        file = request.files['file']
        if not (file and allowed_file(file.filename)):
            flash('Invalid file type')
            return redirect(request.url)

        try:
            filename = generate_unique_filename(secure_filename(file.filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("Invalid image file")

            result = process_image(image, app.config['CROPPED_FOLDER'], filename)
            result['original_image'] = filepath

            return render_template('results.html', 
                               filename=filename, 
                               result=result)

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            flash(f'Error: {str(e)}')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        image_data = request.form.get('image_data')
        if not image_data:
            flash('No image data received')
            return redirect(url_for('capture'))

        try:
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            if len(image_bytes) > 12 * 1024 * 1024:
                flash('Image too large (max 12MB)')
                return redirect(url_for('capture'))

            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                flash('Could not decode image')
                return redirect(url_for('capture'))

            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(filepath, image)

            result = process_image(image, app.config['CROPPED_FOLDER'], filename)
            result['original_image'] = filepath

            return render_template('results.html', filename=filename, result=result)

        except Exception as e:
            logger.error(f"Capture processing error: {e}")
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('capture'))

    return render_template('capture.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
