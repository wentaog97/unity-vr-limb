# Myo EMG Server with Real-time Inference

This server handles both training and real-time inference for EMG gesture recognition using Myo armband data.

## Features

### Training Pipeline (Original)
- Records EMG data from Unity client
- Preprocesses data and extracts features
- Trains Random Forest model
- Exports model to JSON and sends back to client

### Real-time Inference (New)
- Receives EMG data streams from Unity client
- Runs server-side inference using trained models
- Returns pose predictions with confidence scores
- Model selection and management via API

### AprilTag Position Tracking (New)
- Simultaneous camera-based position tracking
- Real-time 6DOF pose estimation (position + rotation)
- Combined EMG + position data in single response
- Configurable target tag ID and camera settings

## Architecture

```
Unity Client (VR Headset)        Python Server
     │                               │
     ├─ Bluetooth EMG Data          ├─ Model Training
     ├─ Stream to Server   ────────> ├─ Real-time Inference  
     ├─ Receive Combined Data <───── ├─ AprilTag Tracking
     └─ Animation + Position        ├─ Model Management
                                    └─ Feature Extraction
```

## Protocol

### Training Commands
```
START_RECORD                    → Begin recording session
timestamp,gesture,EMG,hex_data  → Training data samples
END_RECORD                      → End recording and start training
```

### Inference Commands
```
LIST_MODELS                     → Get available models
SET_MODEL,model_name           → Load specific model for inference
START_INFERENCE                → Begin real-time inference mode
INFERENCE,hex_data             → Send EMG sample for prediction
STOP_INFERENCE                 → End inference mode
```

### AprilTag Commands
```
START_APRILTAG                 → Start camera-based AprilTag tracking
STOP_APRILTAG                  → Stop AprilTag tracking
SET_APRILTAG_ID,tag_id         → Set target AprilTag ID to track
GET_APRILTAG_STATUS            → Request current tracking status
```

### Server Responses
```
MODELS model1,model2,model3    → Available models list
ACK MODEL_SET model_name       → Model successfully loaded
PREDICTION pose confidence     → Inference result (legacy)
PREDICTION_DATA {json}         → Combined EMG + position data (new)
APRILTAG_STATUS {json}         → AprilTag tracking status
ERROR error_message            → Error occurred
```

## Setup and Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Server
```bash
python network_server.py --host 0.0.0.0 --port 5005
```

### 3. Unity Client Setup
- Use `ServerEMGInterpreter` instead of `LocalEMGInterpreter`
- Add `ServerInferenceController` for UI management
- Connect to server IP and port

### 4. Training Workflow
1. Connect Unity client to server
2. Start recording with gesture labels
3. Perform gestures while recording
4. Stop recording - server automatically trains model
5. Model is saved and sent back to client

### Manual Training (Guided Session)
Run the full guidance flow locally (rest/open/clench × three orientations) and train a fresh bundle in one go:

```bash
python manual_train.py --session-id my_custom_session
```

- Auto-discovers a Myo if no address is specified, then walks you through each prompted pose with countdowns and rest windows identical to the server.
- Captured samples are saved as `recordings/<session>_samples.csv`, and the resulting bundle/manifest/metrics land in `models/` (override with `--recordings-dir` / `--models-dir`).
- Use `-p/--path recordings/myo_raw_session_*_samples.csv` to merge in existing CSVs *in addition to* the guided capture, or add `--skip-capture` if you only want to retrain from archived data.
- `--feature-csv` exports the engineered sliding-window table for inspection.

The outputs are drop-in compatible with the server’s `LIST_MODELS` / `SET_MODEL` flow.

> **Heads-up:** `manual_train_recordings.py` now just forwards to `manual_train.py` and prints a deprecation notice.

### Manual Recording & Standalone Inference

You can work entirely from the command line without Unity by using the helper scripts below. Both utilities auto-discover the first Myo armband when no address is provided and share the same CSV/model formats as the server.

#### Run the guided recorder only

```bash
python manual_recording.py --session-id myo_guided_session
```

- Walks you through rest/open/clench in pronation → neutral → supination, holding each pose for 5 seconds with short no-recording rest windows in between (identical to the server’s prompts).
- Displays countdowns and prompts in the terminal so you know what to perform next.
- Outputs `recordings/myo_raw_session_<timestamp>_samples.csv` (or use `--output-dir` / `--session-id` to customize the name/location).
- Use this when you want to capture a full dataset without immediately training; otherwise run `manual_train.py` to capture *and* train in one step.

#### Run live inference

```bash
python manual_inference.py -p models/session_my_custom_session.pkl
```

- Loads the selected bundle, opens a Myo stream, and prints live gesture/orientation predictions with confidence scores and regression outputs.
- Supports `--step-size`, `--print-interval`, and `--max-seconds` for additional control.

### 5. Inference Workflow
1. Connect Unity client to server
2. Request model list: `LIST_MODELS`
3. Select model: `SET_MODEL,model_name`
4. Start inference: `START_INFERENCE`
5. Stream EMG data for real-time predictions
6. Stop when done: `STOP_INFERENCE`

### 6. AprilTag + EMG Combined Workflow
1. Connect Unity client to server
2. Start AprilTag tracking: `START_APRILTAG`
3. Set target tag ID: `SET_APRILTAG_ID,0` (for tag ID 0)
4. Load EMG model: `SET_MODEL,my_gesture_model`
5. Start combined tracking: `START_INFERENCE`
6. Receive `PREDICTION_DATA` with both gesture and position
7. Stop when done: `STOP_INFERENCE` and `STOP_APRILTAG`

## Unity Scripts

### New Components
- **ServerEMGInterpreter**: Handles server-side inference streaming + AprilTag data
- **ServerInferenceController**: UI controller for model selection and AprilTag control
- **NetworkBridgeClient**: Enhanced with prediction and AprilTag event callbacks
- **EMGAprilTagDemo**: Complete demo script showing combined EMG + position tracking

### Migration from Local Inference
Replace:
```csharp
// Old local inference
LocalEMGInterpreter localInterpreter;
localInterpreter.StartInterpretation();
```

With:
```csharp
// New server inference + AprilTag tracking
ServerEMGInterpreter serverInterpreter;
serverInterpreter.SetModel("my_model");
serverInterpreter.SetAprilTagId(0);  // Set target tag ID
serverInterpreter.StartAprilTagTracking();  // Start position tracking
serverInterpreter.StartInterpretation();  // Start EMG inference

// Get combined data
var (position, rotation, hasPos) = serverInterpreter.GetLastPosition();
```

## Performance Benefits

- **Faster Inference**: Server-grade CPU vs mobile processor
- **Better Models**: Can use larger, more complex models
- **Centralized Updates**: Update models without client deployment
- **Scalability**: Multiple clients can use same server
- **Resource Efficiency**: Client device saves battery and compute

## File Structure

```
myo-server/
├── network_server.py          # Main server with training + inference
├── feature_extraction.py      # EMG feature extraction functions
├── train_rf_fe.py             # Random Forest training script
├── data_preprocessing.py      # Data cleaning and preprocessing
├── export_rf_json.py          # Model export for Unity
├── requirements.txt           # Python dependencies
├── models/                    # Trained model storage
├── recordings/                # Training data storage
└── README.md                  # This file
```

## Troubleshooting

### Connection Issues
- Check firewall settings on server machine
- Ensure Unity client has correct server IP
- Verify port 5005 is not blocked

### Model Loading Errors
- Ensure model files exist in `models/` directory
- Check model file naming conventions
- Verify model was trained with compatible feature extraction

### Inference Problems
- Check EMG data format (32-character hex strings)
- Verify model is loaded before starting inference
- Monitor server logs for detailed error messages

## Development

### Adding New Features
- Extend protocol in `network_server.py`
- Update Unity `NetworkBridgeClient` message handling
- Add corresponding UI elements in `ServerInferenceController`

### Custom Models
- Modify `train_rf_fe.py` for different algorithms
- Update `extract_features_from_window()` for new features
- Ensure compatibility with Unity inference expectations