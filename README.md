# Traffic-Analysis

A comprehensive Gradio application that combines Vision-Language Models (VLLMs) with computer vision techniques to analyze traffic scenes and detect license plates with high accuracy.

## 🎯 Overview

This application integrates multiple state-of-the-art AI models to provide:

1. **Traffic Scene Description** - Using LLaVA-NeXT for comprehensive scene understanding
2. **License Plate Detection** - Using YOLOv11 for accurate plate localization
3. **Text Extraction** - Using PaddleOCR with advanced preprocessing for text recognition
4. **Structured Output** - JSON format combining all analysis results

## ✨ Features

### Core Capabilities
- 🔍 **Multi-modal Analysis**: Combines vision and language understanding
- 🎯 **Accurate Detection**: YOLOv11-based license plate detection
- 📝 **Robust OCR**: PaddleOCR with preprocessing and confidence filtering
- ⚙️ **Parameter Control**: Adjustable thresholds and generation parameters
- 🚀 **Optimized Performance**: Memory-efficient model loading with quantization
- 📊 **Structured Output**: Task-compliant JSON format

### Adjustable Parameters
- **YOLO Confidence Threshold** (0.1-1.0): Controls detection sensitivity
- **OCR Confidence Threshold** (0.0-1.0): Filters low-quality text recognition
- **VLLM Temperature** (0.1-2.0): Controls creativity/randomness in descriptions
- **VLLM Top-p** (0.1-1.0): Controls diversity vs focus in language generation

## 🛠️ Setup Instructions

### Environment Requirements
- **Platform**: Kaggle
- **Python**: 3.8+
- **CUDA**: Compatible GPU for optimal performance (Kaggle typically provides a T4 or P100 GPU)

### Step 1: Install Dependencies

```bash
# Install packages
!pip install gradio ultralytics paddlepaddle paddleocr transformers torch torchvision accelerate bitsandbytes opencv-python pillow numpy
```

### Step 2: Upload Model Files
Upload custom YOLOv11 model trained for license plate detection
- `/kaggle/input/license_plate_detect_yolo11/pytorch/default/1/best.pt`
Kaggle model link: https://www.kaggle.com/models/suhailaaboubakr/license_plate_detect_yolo11/

## 🏗️ Model Architecture

### 1. Scene Understanding: LLaVA-NeXT
- **Model**: `llava-hf/llava-v1.6-mistral-7b-hf`
- **Quantization**: 4-bit with BitsAndBytesConfig
- **Purpose**: Generate comprehensive traffic scene descriptions
- **Optimization**: Memory-efficient loading with device mapping

### 2. License Plate Detection: YOLOv11
- **Model**: Custom trained `best.pt`
- **Input Size**: 640x640 optimized
- **Purpose**: Detect and localize license plates in images
- **Output**: Bounding boxes with confidence scores

### 3. Text Recognition: PaddleOCR
- **Language**: English optimized
- **Features**: Angle classification, text detection & recognition
- **Preprocessing**: CLAHE enhancement, noise reduction, thresholding
- **Purpose**: Extract text from detected license plate regions

## 📖 Usage Examples

### Basic Usage

1. **Upload Image**: Select a traffic scene image
2. **Adjust Parameters** (optional):
   - YOLO Confidence: 0.5 (default)
   - OCR Confidence: 0.1 (default)
   - Temperature: 0.7 (default)
   - Top-p: 0.9 (default)
3. **Click Submit**: Process the image
4. **Review Results**: Scene description, plate details, and JSON output

## Sample Test
### Test Image 1
<img width="1605" height="806" alt="test_image1" src="https://github.com/user-attachments/assets/3fad5398-f119-4989-81d5-58102cc792a8" />

### Test Image 2
<img width="1591" height="839" alt="test_image2" src="https://github.com/user-attachments/assets/b89a4b6d-78e4-4383-a9c2-fe4825e99737" />

### Test Image 3
<img width="1604" height="808" alt="test_image3" src="https://github.com/user-attachments/assets/856304d5-e8d3-47f4-847e-e8f1fd94b532" />

## 🎛️ Parameter Tuning Guide

### YOLO Confidence Threshold
- **Low (0.1-0.3)**: Detects more plates, higher false positive rate
- **Medium (0.4-0.6)**: Balanced detection, recommended for most cases
- **High (0.7-1.0)**: Only high-confidence detections, may miss some plates

### OCR Confidence Threshold
- **Very Low (0.0-0.1)**: Accept all OCR results, may include noise
- **Low (0.1-0.3)**: Accept most readable text, some false readings
- **Medium (0.3-0.6)**: Good balance of accuracy and recall
- **High (0.6-1.0)**: Only high-quality text, may miss valid plates

### VLLM Temperature
- **Low (0.1-0.3)**: More focused, factual descriptions
- **Medium (0.4-0.6)**: Balanced creativity and accuracy
- **High (0.7-1.0)**: More creative, potentially less accurate

### VLLM Top-p
- **Low (0.1-0.5)**: Conservative vocabulary, more predictable
- **Medium (0.6-0.8)**: Balanced diversity
- **High (0.9-1.0)**: Maximum vocabulary diversity

## 📊 JSON Output Format

The application outputs a structured JSON following the task specifications:

```json
{
  "scene_description": "Detailed analysis of the traffic scene including vehicles, infrastructure, conditions, and safety considerations",
  "total_plates_detected": ,
  "license_plates": [
    {
      "bounding_box": [150, 200, 250, 240],
      "text": "License Plate Text",
      "confidence":
    }
  ],
  'parameters_used': {
      'yolo_confidence_threshold': ,
      'ocr_confidence_threshold': ,
      'vllm_temperature': ,
      'vllm_top_p': 
  },
}
```

## 🎯 Prompt Engineering Rationale

### Scene Description Prompt Design

The VLLM prompt is carefully crafted to extract maximum traffic-relevant information:

```
Analyze this traffic scene in detail. Describe:
1. Types of vehicles present (cars, trucks, motorcycles, etc.)
2. Traffic signs, signals, and road markings visible
3. Road conditions and infrastructure
4. Weather and lighting conditions
5. Overall traffic flow and density
6. Any notable safety considerations or hazards
```

**Rationale**:
- **Structured approach**: Numbered points ensure comprehensive coverage
- **Traffic-focused**: Specifically targets transportation elements
- **Safety-oriented**: Includes hazard identification
- **Detailed yet concise**: Balances thoroughness with readability

### Parameter Choices

**Default Temperature (0.7)**:
- Balances factual accuracy with descriptive richness
- Avoids overly repetitive descriptions
- Maintains focus on observable elements

**Default Top-p (0.9)**:
- Allows diverse vocabulary while maintaining coherence
- Prevents overly conservative language choices
- Enables detailed technical descriptions

### Performance Optimization

1. **Memory Management**:
   - Use model quantization (4-bit enabled by default)
   - Clear GPU cache between runs if needed
   - Monitor VRAM usage

2. **Speed Optimization**:
   - Resize large images before processing
   - Use appropriate batch sizes
   - Enable half-precision when supported

3. **Accuracy Improvement**:
   - Use high-quality input images
   - Adjust confidence thresholds based on use case
   - Consider image preprocessing for difficult lighting

## 📈 Expected Performance

### Typical Results
- **Scene Description**: Comprehensive 150-250 word analysis
- **Detection Accuracy**: 85-95% for clearly visible plates
- **OCR Accuracy**: 80-90% for standard license plates
- **Processing Time**: 10-30 seconds per image on T4 GPU

### Limitations
- Performance depends on image quality and lighting
- OCR accuracy varies with plate condition and angle
- Complex scenes may require parameter adjustment
- GPU memory limits maximum image resolution
