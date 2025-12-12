using System;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

/// <summary>
/// Server-side EMG interpreter that streams EMG data to server and receives predictions.
/// Replaces LocalEMGInterpreter for server-side inference architecture.
/// </summary>
public class ServerEMGInterpreter : MonoBehaviour 
{
    [Header("References")]
    // Note (Phase 2): MyoBluetoothManager no longer used - BLE handled on server
    public NetworkBridgeClient netClient;
    public TextMeshProUGUI poseText;
    
    [Header("Model Management")]
    [Tooltip("Model name to use for inference (set via SetModel or from server list)")]
    public string activeModelName = "";
    
    [Header("Animator Bridge")]
    public GestureController gestureController;
    public float smoothTime = 0.1f;

    [Header("Wrist Control")]
    [Tooltip("Apply orientation outputs to the wrist/forearm twist controller.")]
    public bool driveForearmTwist = true;
    public WristOffsetsController wristController;
    [Tooltip("Forearm rotation (deg) when orientation = pronation (-1).")]
    public float pronationTwistDegrees = -120f;
    [Tooltip("Forearm rotation (deg) when orientation = neutral (0).")]
    public float neutralTwistDegrees = 60f;
    [Tooltip("Forearm rotation (deg) when orientation = supination (+1).")]
    public float supinationTwistDegrees = 240f;

    [Header("Orientation Filtering")]
    [Tooltip("Smooth incoming orientation value instead of applying raw predictions.")]
    public bool enableOrientationFilter = true;
    [Tooltip("Time (seconds) for SmoothDamp on orientation value. 0 = snap.")]
    public float orientationSmoothTime = 0.12f;
    [Tooltip("Ignore orientation deltas smaller than this normalized amount (0-2 range).")]
    public float orientationDeadband = 0.02f;
    [Tooltip("Minimum orientation confidence required to accept an update (0-1).")]
    [Range(0f, 1f)] public float orientationConfidenceThreshold = 0.1f;

    [Header("Gesture Filtering")]
    [Tooltip("Smooth/hold gesture outputs instead of applying every frame.")]
    public bool enableGestureFilter = true;
    [Tooltip("Time (seconds) for SmoothDamp on gesture blend. 0 = snap.")]
    public float gestureSmoothTime = 0.08f;
    [Tooltip("Ignore gesture blend deltas smaller than this amount (range -1 to 1).")]
    public float gestureDeadband = 0.02f;
    [Tooltip("Minimum gesture confidence required to accept an update (0-1).")]
    [Range(0f, 1f)] public float gestureConfidenceThreshold = 0.05f;
    [Tooltip("Number of consecutive frames a new gesture label must appear before we switch to it.")]
    [Min(1)] public int gestureRequiredStableFrames = 2;
    
    // Runtime state
    private bool inferenceRunning = false;
    private readonly Dictionary<string, float> lastPredictions = new();
    private float currentBlend;
    private float blendVelocity;
    private List<string> availableModels = new();
    
    // Thread-safe counters for debug log throttling (since Time.frameCount can't be used on background threads)
    private int parseCallCount = 0;
    private int getPositionCallCount = 0;
    
    // Prediction display
    private string lastPose = "--";
    private string lastGestureLabel = "--";
    private string lastOrientationLabel = "--";
    private float lastConfidence = 0f;
    private float lastGestureBlend = 0f;
    private float lastOrientationValue = 0f;
    private bool hasOrientationValue = false;
    private bool hasGestureBlend = false;
    private float filteredOrientationValue = 0f;
    private float orientationFilterVelocity = 0f;
    private bool hasFilteredOrientation = false;
    private string filteredGestureLabel = "--";
    private float filteredGestureBlend = 0f;
    private float gestureFilterVelocity = 0f;
    private bool hasFilteredGesture = false;
    private string gestureCandidateLabel = "--";
    private int gestureStabilityFrames = 0;
    
    // Multi-joint position tracking data
    private Dictionary<int, JointPositionData> jointPositions = new Dictionary<int, JointPositionData>();
    
    [System.Serializable]
    public class JointPositionData
    {
        public Vector3 tagPosition;
        public Vector3 jointCenter;
        public Vector3 rotation;
        public float timestamp;
        public float offsetApplied;
    }
    
    private void Awake()
    {
        // Safety: Ensure dictionaries are initialized (Unity serialization can sometimes clear them)
        if (jointPositions == null)
        {
            jointPositions = new Dictionary<int, JointPositionData>();
        }
        
        // Find components if not assigned
        if (!netClient) netClient = FindFirstObjectByType<NetworkBridgeClient>();
        if (!gestureController) gestureController = FindFirstObjectByType<GestureController>();
        if (!wristController) wristController = FindFirstObjectByType<WristOffsetsController>();
        
        // Subscribe to network messages for predictions
        if (netClient) RegisterNetworkCallbacks();
    }
    
    private void OnDestroy()
    {
        if (netClient)
        {
            netClient.PredictionReceived -= OnServerPrediction;
            netClient.PredictionDataReceived -= OnServerPredictionData;
            netClient.ModelListReceived -= OnServerModelList;
            netClient.ModelSetConfirmed -= OnModelSetConfirmation;
            netClient.AprilTagStatusReceived -= OnAprilTagStatus;
            netClient.ErrorReceived -= OnServerError;
        }
        
        StopInterpretation();
    }
    
    private void RegisterNetworkCallbacks()
    {
        if (netClient)
        {
            netClient.PredictionReceived += OnServerPrediction;
            netClient.PredictionDataReceived += OnServerPredictionData;
            netClient.ModelListReceived += OnServerModelList;
            netClient.ModelSetConfirmed += OnModelSetConfirmation;
            netClient.AprilTagStatusReceived += OnAprilTagStatus;
            netClient.ErrorReceived += OnServerError;
        }
    }
    
    private void OnServerError(string error)
    {
        Debug.LogError($"[ServerEMG] Server error: {error}");
        if (poseText) poseText.text = $"Error: {error}";
    }
    
    // ==================== PUBLIC API ====================
    
    /// <summary>
    /// Request list of available models from server
    /// </summary>
    public void RequestModelList()
    {
        if (netClient && netClient.IsConnected)
        {
            netClient.QueueMessage("LIST_MODELS");
            Debug.Log("[ServerEMG] Requesting model list from server");
        }
        else
        {
            Debug.LogWarning("[ServerEMG] Not connected to server");
        }
    }
    
    /// <summary>
    /// Set the active model on the server
    /// </summary>
    public void SetModel(string modelName)
    {
        if (netClient && netClient.IsConnected)
        {
            netClient.QueueMessage($"SET_MODEL,{modelName}");
            Debug.Log($"[ServerEMG] Setting server model to: {modelName}");
        }
        else
        {
            Debug.LogWarning("[ServerEMG] Not connected to server");
        }
    }
    
    /// <summary>
    /// Start real-time inference streaming
    /// </summary>
    public void StartInterpretation()
    {
        if (netClient && netClient.IsConnected)
        {
            netClient.QueueMessage("START_INFERENCE");
            inferenceRunning = true;
            lastPredictions.Clear();
            Debug.Log("[ServerEMG] Starting server-side inference");
            
            // Update UI
            if (poseText) poseText.text = "Starting inference...";
        }
        else
        {
            Debug.LogWarning("[ServerEMG] Not connected to server");
            if (poseText) poseText.text = "Not connected";
        }
    }
    
    /// <summary>
    /// Stop real-time inference streaming
    /// </summary>
    public void StopInterpretation()
    {
        if (netClient && netClient.IsConnected && inferenceRunning)
        {
            netClient.QueueMessage("STOP_INFERENCE");
        }
        
        inferenceRunning = false;
        lastPredictions.Clear();
        hasOrientationValue = false;
        lastOrientationValue = 0f;
        filteredOrientationValue = 0f;
        orientationFilterVelocity = 0f;
        hasFilteredOrientation = false;
        hasGestureBlend = false;
        lastGestureBlend = 0f;
        filteredGestureBlend = 0f;
        gestureFilterVelocity = 0f;
        hasFilteredGesture = false;
        filteredGestureLabel = "--";
        gestureCandidateLabel = "--";
        gestureStabilityFrames = 0;
        Debug.Log("[ServerEMG] Stopped server-side inference");
        
        // Update UI
        if (poseText) poseText.text = "--";
    }
    
    /// <summary>
    /// Check if inference is currently running
    /// </summary>
    public bool IsRunning => inferenceRunning;
    
    /// <summary>
    /// Get list of available models (call RequestModelList first)
    /// </summary>
    public List<string> GetAvailableModels() => new(availableModels);
    
    /// <summary>
    /// Start AprilTag tracking on the server
    /// </summary>
    public void StartAprilTagTracking()
    {
        if (netClient && netClient.IsConnected)
        {
            netClient.QueueMessage("START_APRILTAG");
            Debug.Log("[ServerEMG] Starting AprilTag tracking on server");
        }
        else
        {
            Debug.LogWarning("[ServerEMG] Not connected to server");
        }
    }
    
    /// <summary>
    /// Stop AprilTag tracking on the server
    /// </summary>
    public void StopAprilTagTracking()
    {
        if (netClient && netClient.IsConnected)
        {
            netClient.QueueMessage("STOP_APRILTAG");
            Debug.Log("[ServerEMG] Stopping AprilTag tracking on server");
        }
    }
    
    /// <summary>
    /// Add a joint to track with AprilTag
    /// </summary>
    /// <param name="tagId">AprilTag ID to track</param>
    /// <param name="offsetDistance">Distance in meters to offset from tag surface to joint center</param>
    public void AddJoint(int tagId, float offsetDistance)
    {
        Debug.Log($"[ServerEMG] AddJoint called: tag_id={tagId}, offset={offsetDistance}m");
        
        if (!netClient)
        {
            Debug.LogError("[ServerEMG] netClient is null!");
            return;
        }
        
        if (!netClient.IsConnected)
        {
            Debug.LogWarning($"[ServerEMG] Not connected to server (IsConnected={netClient.IsConnected})");
            return;
        }
        
        string command = $"ADD_APRILTAG_JOINT,{tagId},{offsetDistance}";
        Debug.Log($"[ServerEMG] Sending command: {command}");
        netClient.QueueMessage(command);
        Debug.Log($"[ServerEMG] Command queued successfully");
    }
    
    /// <summary>
    /// Remove a joint from tracking
    /// </summary>
    public void RemoveJoint(int tagId)
    {
        if (netClient && netClient.IsConnected)
        {
            netClient.QueueMessage($"REMOVE_APRILTAG_JOINT,{tagId}");
            Debug.Log($"[ServerEMG] Removing joint: tag_id={tagId}");
        }
    }
    
    /// <summary>
    /// Clear all tracked joints
    /// </summary>
    public void ClearAllJoints()
    {
        if (netClient && netClient.IsConnected)
        {
            netClient.QueueMessage("CLEAR_APRILTAG_JOINTS");
            Debug.Log("[ServerEMG] Clearing all joints");
        }
    }
    
    /// <summary>
    /// Request list of tracked joints from server
    /// </summary>
    public void RequestJointList()
    {
        if (netClient && netClient.IsConnected)
        {
            netClient.QueueMessage("LIST_APRILTAG_JOINTS");
        }
    }
    
    /// <summary>
    /// Request AprilTag status from server
    /// </summary>
    public void RequestAprilTagStatus()
    {
        if (netClient && netClient.IsConnected)
        {
            netClient.QueueMessage("GET_APRILTAG_STATUS");
        }
    }
    
    /// <summary>
    /// Get joint center position for a specific tag ID
    /// </summary>
    public (Vector3 jointCenter, Vector3 rotation, bool hasData) GetJointPosition(int tagId)
    {
        // Safety: Ensure dictionary is initialized (Unity serialization can sometimes clear it)
        if (jointPositions == null)
        {
            jointPositions = new Dictionary<int, JointPositionData>();
            Debug.LogWarning("[ServerEMG] jointPositions was null, reinitializing...");
        }
        
        // Increment counter and check if we should log (thread-safe alternative to Time.frameCount)
        getPositionCallCount++;
        bool shouldLog = (getPositionCallCount % 120 == 0);
        
        if (jointPositions.TryGetValue(tagId, out var jointData))
        {
            // Debug: Log occasionally when we successfully return data
            if (shouldLog)
            {
                Debug.Log($"[ServerEMG] GetJointPosition({tagId}): FOUND - pos=({jointData.jointCenter.x:F3}, {jointData.jointCenter.y:F3}, {jointData.jointCenter.z:F3})");
            }
            return (jointData.jointCenter, jointData.rotation, true);
        }
        
        // Debug: Log when requested tag is not found
        if (shouldLog)
        {
            string availableTags = jointPositions.Count > 0 ? string.Join(", ", jointPositions.Keys) : "none";
            Debug.Log($"[ServerEMG] GetJointPosition({tagId}): NOT FOUND. Available tags: [{availableTags}]");
        }
        return (Vector3.zero, Vector3.zero, false);
    }
    
    /// <summary>
    /// Get all tracked joint positions
    /// </summary>
    public Dictionary<int, JointPositionData> GetAllJointPositions()
    {
        // Safety: Ensure dictionary is initialized
        if (jointPositions == null)
        {
            jointPositions = new Dictionary<int, JointPositionData>();
            Debug.LogWarning("[ServerEMG] jointPositions was null in GetAllJointPositions, reinitializing...");
        }
        
        return new Dictionary<int, JointPositionData>(jointPositions);
    }
    
    /// <summary>
    /// Check if a specific joint has data
    /// </summary>
    public bool HasJointData(int tagId)
    {
        return jointPositions.ContainsKey(tagId);
    }
    
    /// <summary>
    /// [DEPRECATED] Use GetJointPosition(tagId) instead
    /// Get last known position data (legacy single-tag support)
    /// </summary>
    [System.Obsolete("Use GetJointPosition(tagId) for multi-joint support")]
    public (Vector3 position, Vector3 rotation, bool hasData) GetLastPosition()
    {
        // Return first available joint for backwards compatibility
        if (jointPositions.Count > 0)
        {
            var first = jointPositions.Values.GetEnumerator();
            if (first.MoveNext())
            {
                var data = first.Current;
                return (data.jointCenter, data.rotation, true);
            }
        }
        return (Vector3.zero, Vector3.zero, false);
    }
    
    // ==================== EMG DATA STREAMING ====================
    // NOTE (Phase 2): EMG data is now streamed directly from BLE on the server side.
    // Unity no longer forwards EMG data. The server automatically streams EMG when 
    // START_INFERENCE is called and a Myo is connected.
    //
    // MULTI-ARMBAND SUPPORT (Phase 2+):
    // - Multiple armbands can be connected simultaneously
    // - Current implementation: Server uses first connected armband for inference
    // - Future (Phase 3): Bilateral inference combining data from all connected armbands
    //   Would require: 32-channel models (16ch × 2 armbands) or parallel inference pipelines
    //
    // For now: Inference uses one armband at a time (the first connected one)
    // Recording: Can record from specific selected armband
    
    // OnEmgData method removed - no longer needed!
    
    // ==================== SERVER RESPONSE HANDLING ====================
    
    private void Update()
    {
        // Handle server responses
        // Note: This is a simplified approach. In a real implementation, you might want
        // to modify NetworkBridgeClient to expose a callback system for specific message types
        
        // Update blend animation
        if (gestureController && !string.IsNullOrEmpty(lastPose))
        {
            string poseKey = string.IsNullOrEmpty(lastGestureLabel) ? lastPose : lastGestureLabel;
            float target = hasGestureBlend ? lastGestureBlend : MapPoseToBlend(poseKey);
            currentBlend = Mathf.SmoothDamp(currentBlend, target, ref blendVelocity, smoothTime);
            gestureController.SetGestureBlend(currentBlend);
        }

        // Update forearm twist
        if (driveForearmTwist && wristController && hasOrientationValue)
        {
            float targetDegrees;
            if (lastOrientationValue >= 0f)
            {
                targetDegrees = Mathf.Lerp(neutralTwistDegrees, supinationTwistDegrees, lastOrientationValue);
            }
            else
            {
                targetDegrees = Mathf.Lerp(neutralTwistDegrees, pronationTwistDegrees, -lastOrientationValue);
            }

            if (wristController != null)
            {
                targetDegrees = Mathf.Clamp(targetDegrees, wristController.supClampDeg.x, wristController.supClampDeg.y);
            }

            wristController.supInput = targetDegrees;
        }
        
        // Update UI
        UpdateUI();
    }
    
    /// <summary>
    /// Call this method when receiving a PREDICTION message from server
    /// Format: "PREDICTION pose confidence"
    /// </summary>
    public void OnServerPrediction(string pose, float confidence)
    {
        lastPose = pose;
        lastConfidence = confidence;
        
        // Store for smoothing/filtering if needed
        lastPredictions[pose] = confidence;
        
        Debug.Log($"[ServerEMG] Received prediction: {pose} ({confidence:P1})");
    }
    
    /// <summary>
    /// Call this method when receiving a PREDICTION_DATA message from server
    /// Format: "PREDICTION_DATA {json}" - includes EMG prediction + multi-joint AprilTag positions
    /// </summary>
    public void OnServerPredictionData(string jsonData)
    {
        try
        {
            // Try to parse EMG prediction data (may not exist for JOINT_POSITIONS responses)
            var basicData = JsonUtility.FromJson<BasicPredictionData>(jsonData);
            
            // Only update pose/confidence if they exist (not null/empty)
                if (!string.IsNullOrEmpty(basicData.pose))
                {
                    lastPose = basicData.pose;
                    lastConfidence = basicData.confidence;
                    lastPredictions[basicData.pose] = basicData.confidence;
                    Debug.Log($"[ServerEMG] Prediction: {basicData.pose} ({basicData.confidence:P1})");

                    string incomingGestureLabel = null;
                    if (!string.IsNullOrEmpty(basicData.gesture))
                    {
                        incomingGestureLabel = basicData.gesture;
                    }
                    else if (string.IsNullOrEmpty(lastGestureLabel))
                    {
                        incomingGestureLabel = basicData.pose;
                    }
                    else
                    {
                        incomingGestureLabel = lastGestureLabel;
                    }

                    float rawGestureBlend = Mathf.Clamp(basicData.gesture_blend, -1f, 1f);
                    float gestureConf = Mathf.Clamp01(basicData.gesture_confidence);
                    bool acceptGestureUpdate = gestureConf >= gestureConfidenceThreshold || !hasFilteredGesture;

                    if (!enableGestureFilter || gestureSmoothTime <= 0f)
                    {
                        if (acceptGestureUpdate)
                        {
                            filteredGestureLabel = incomingGestureLabel;
                            filteredGestureBlend = rawGestureBlend;
                            gestureFilterVelocity = 0f;
                            hasFilteredGesture = true;
                        }
                    }
                    else if (acceptGestureUpdate)
                    {
                        if (incomingGestureLabel != gestureCandidateLabel)
                        {
                            gestureCandidateLabel = incomingGestureLabel;
                            gestureStabilityFrames = 0;
                        }
                        gestureStabilityFrames++;
                        bool labelStable = gestureStabilityFrames >= gestureRequiredStableFrames || !hasFilteredGesture;
                        if (labelStable)
                        {
                            filteredGestureLabel = incomingGestureLabel;
                        }

                        if (!hasFilteredGesture)
                        {
                            filteredGestureBlend = rawGestureBlend;
                            gestureFilterVelocity = 0f;
                        }
                        else
                        {
                            float delta = Mathf.Abs(rawGestureBlend - filteredGestureBlend);
                            if (delta <= gestureDeadband)
                            {
                                filteredGestureBlend = rawGestureBlend;
                            }
                            else
                            {
                                filteredGestureBlend = Mathf.SmoothDamp(
                                    filteredGestureBlend,
                                    rawGestureBlend,
                                    ref gestureFilterVelocity,
                                    gestureSmoothTime);
                            }
                        }

                        hasFilteredGesture = true;
                    }

                    lastGestureLabel = filteredGestureLabel;
                    lastGestureBlend = filteredGestureBlend;
                    hasGestureBlend = hasFilteredGesture;

                    if (!string.IsNullOrEmpty(basicData.orientation))
                    {
                        lastOrientationLabel = basicData.orientation;
                        float rawOrientation = Mathf.Clamp(basicData.orientation_value, -1f, 1f);
                        float orientationConf = Mathf.Clamp01(basicData.orientation_confidence);
                        bool acceptUpdate = orientationConf >= orientationConfidenceThreshold || !hasFilteredOrientation;

                        if (!enableOrientationFilter || orientationSmoothTime <= 0f)
                        {
                            if (acceptUpdate)
                            {
                                filteredOrientationValue = rawOrientation;
                                orientationFilterVelocity = 0f;
                                hasFilteredOrientation = true;
                            }
                        }
                        else if (acceptUpdate)
                        {
                            if (!hasFilteredOrientation)
                            {
                                filteredOrientationValue = rawOrientation;
                                orientationFilterVelocity = 0f;
                            }

                            float delta = Mathf.Abs(rawOrientation - filteredOrientationValue);
                            if (delta <= orientationDeadband)
                            {
                                filteredOrientationValue = rawOrientation;
                            }
                            else
                            {
                                filteredOrientationValue = Mathf.SmoothDamp(
                                    filteredOrientationValue,
                                    rawOrientation,
                                    ref orientationFilterVelocity,
                                    orientationSmoothTime);
                            }

                            hasFilteredOrientation = true;
                        }

                        lastOrientationValue = filteredOrientationValue;
                        hasOrientationValue = hasFilteredOrientation;
                    }
                }
            
            // Parse joints data manually (Unity JsonUtility doesn't handle dictionaries)
            ParseJointsData(jsonData);
            
            // Note: Don't log here - logging happens inside ParseJointsData with thread-safe counters
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[ServerEMG] Failed to parse prediction data: {e.Message}");
        }
    }
    
    private void ParseJointsData(string jsonData)
    {
        try
        {
            // Safety: Ensure dictionary is initialized
            if (jointPositions == null)
            {
                jointPositions = new Dictionary<int, JointPositionData>();
                Debug.LogWarning("[ServerEMG] jointPositions was null in ParseJointsData, reinitializing...");
            }
            
            // Increment counter (thread-safe alternative to Time.frameCount)
            parseCallCount++;
            bool shouldLog = (parseCallCount % 60 == 0);
            
            // Find the "joints" section in JSON
            int jointsStart = jsonData.IndexOf("\"joints\":");
            if (jointsStart == -1)
            {
                // No joints data in this response
                return;
            }
            
            jointsStart += 9; // Skip "joints":
            int jointsEnd = jsonData.LastIndexOf('}');
            if (jointsEnd == -1) return;
            
            string jointsSection = jsonData.Substring(jointsStart, jointsEnd - jointsStart);
            
            // Debug: Log the joints section we're about to parse (occasionally)
            if (shouldLog)
            {
                Debug.Log($"[ServerEMG] Parsing joints section (first 200 chars): {jointsSection.Substring(0, Mathf.Min(200, jointsSection.Length))}...");
            }
            
            // Clear old data
            jointPositions.Clear();
            
            // Simple parser for format: "tagId":{"timestamp":...,"joint_center":{...},...}
            // Split by tag IDs (looking for "number":{)
            int pos = 0;
            while (pos < jointsSection.Length)
            {
                // Find next tag ID
                int tagIdStart = jointsSection.IndexOf('"', pos);
                if (tagIdStart == -1) break;
                tagIdStart++;
                
                int tagIdEnd = jointsSection.IndexOf('"', tagIdStart);
                if (tagIdEnd == -1) break;
                
                string tagIdStr = jointsSection.Substring(tagIdStart, tagIdEnd - tagIdStart);
                if (!int.TryParse(tagIdStr, out int tagId))
                {
                    pos = tagIdEnd + 1;
                    continue;
                }
                
                // Find the joint data object for this tag
                int objStart = jointsSection.IndexOf('{', tagIdEnd);
                if (objStart == -1) break;
                
                // Find matching closing brace
                int braceCount = 1;
                int objEnd = objStart + 1;
                while (objEnd < jointsSection.Length && braceCount > 0)
                {
                    if (jointsSection[objEnd] == '{') braceCount++;
                    if (jointsSection[objEnd] == '}') braceCount--;
                    objEnd++;
                }
                
                string jointJson = jointsSection.Substring(objStart, objEnd - objStart);
                
                // Parse the joint data
                var jointData = JsonUtility.FromJson<JointData>("{" + jointJson.Substring(1));
                
                if (jointData != null && jointData.joint_center != null)
                {
                    var posData = new JointPositionData
                    {
                        tagPosition = new Vector3(jointData.tag_position.x, jointData.tag_position.y, jointData.tag_position.z),
                        jointCenter = new Vector3(jointData.joint_center.x, jointData.joint_center.y, jointData.joint_center.z),
                        rotation = new Vector3(jointData.rotation_degrees.roll, jointData.rotation_degrees.pitch, jointData.rotation_degrees.yaw),
                        timestamp = jointData.timestamp,
                        offsetApplied = jointData.offset_applied
                    };
                    
                    jointPositions[tagId] = posData;
                    
                    // Debug: Log when we successfully parse a tag (using counter, not frameCount)
                    if (shouldLog)
                    {
                        Debug.Log($"[ServerEMG] Parsed TagID {tagId}: pos=({posData.jointCenter.x:F3}, {posData.jointCenter.y:F3}, {posData.jointCenter.z:F3})");
                    }
                }
                
                pos = objEnd;
            }
            
            // Debug: Log what tags we have after parsing (using counter, not frameCount)
            if (jointPositions.Count > 0 && shouldLog)
            {
                string tagList = string.Join(", ", jointPositions.Keys);
                Debug.Log($"[ServerEMG] After parsing, jointPositions contains {jointPositions.Count} tags: [{tagList}]");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[ServerEMG] Failed to parse joints data: {e.Message}");
            Debug.LogWarning($"[ServerEMG] Stack trace: {e.StackTrace}");
        }
    }
    
    /// <summary>
    /// Call this method when receiving AprilTag status from server
    /// </summary>
    public void OnAprilTagStatus(string statusJson)
    {
        try
        {
            var status = JsonUtility.FromJson<AprilTagStatus>(statusJson);
            string joints = status.tracked_joints != null && status.tracked_joints.Length > 0 
                ? string.Join(", ", status.tracked_joints) 
                : "None";
            Debug.Log($"[ServerEMG] AprilTag status - Running: {status.running}, Tracked Joints: {joints}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[ServerEMG] Failed to parse AprilTag status: {e.Message}");
        }
    }
    
    /// <summary>
    /// Call this method when receiving a MODELS message from server
    /// Format: "MODELS model1,model2,model3" or "MODELS NONE"
    /// </summary>
    public void OnServerModelList(string modelListStr)
    {
        availableModels.Clear();
        
        if (modelListStr != "NONE" && !string.IsNullOrEmpty(modelListStr))
        {
            string[] models = modelListStr.Split(',');
            availableModels.AddRange(models);
        }
        
        Debug.Log($"[ServerEMG] Available models: {string.Join(", ", availableModels)}");
    }
    
    /// <summary>
    /// Call this method when receiving model set confirmation
    /// Format: "ACK MODEL_SET modelName"
    /// </summary>
    public void OnModelSetConfirmation(string modelName)
    {
        activeModelName = modelName;
        Debug.Log($"[ServerEMG] Active model set to: {modelName}");
    }
    
    // ==================== UI AND ANIMATION ====================
    
    private void UpdateUI()
    {
        if (poseText)
        {
            if (inferenceRunning)
            {
                if (!string.IsNullOrEmpty(lastPose))
                {
                    string jointsInfo = "";
                    if (jointPositions.Count > 0)
                    {
                        jointsInfo = $"\n\nTracked Joints: {jointPositions.Count}";
                        foreach (var kvp in jointPositions)
                        {
                            int tagId = kvp.Key;
                            var data = kvp.Value;
                            jointsInfo += $"\nTag {tagId}: ({data.jointCenter.x:F3}, {data.jointCenter.y:F3}, {data.jointCenter.z:F3})";
                        }
                    }
                    else
                    {
                        jointsInfo = "\n\nJoints: No AprilTags detected";
                    }
                    
                    string gestureLine = string.IsNullOrEmpty(lastGestureLabel) ? lastPose : lastGestureLabel;
                    string orientationLine = string.IsNullOrEmpty(lastOrientationLabel) ? "--" : lastOrientationLabel;
                    poseText.text =
                        $"Gesture: {gestureLine}\n" +
                        $"Orientation: {orientationLine}\n" +
                        $"Pose Label: {lastPose}\n" +
                        $"Confidence: {lastConfidence:P1}\n" +
                        $"Gesture Blend: {lastGestureBlend:F2}\n" +
                        $"Orientation Value: {lastOrientationValue:F2}\n" +
                        $"Blend (smoothed): {currentBlend:F2}\n" +
                        $"Model: {activeModelName}{jointsInfo}";
                }
                else
                {
                    poseText.text = $"Waiting for predictions...\nModel: {activeModelName}";
                }
            }
            else
            {
                poseText.text = "Inference stopped";
            }
        }
    }
    
    private float MapPoseToBlend(string pose)
    {
        pose = pose.ToLowerInvariant();
        
        // Map poses to blend values for animation
        if (pose.Contains("clench") || pose.Contains("fist") || pose.Contains("close")) return -1f;
        if (pose.Contains("rest") || pose.Contains("neutral")) return 0f;
        if (pose.Contains("open") || pose.Contains("extend")) return 1f;
        
        // Default to neutral so the animator does not hold stale data
        return 0f;
    }
}

// ==================== JSON DATA CLASSES ====================

[System.Serializable]
public class BasicPredictionData
{
    public string type;
    public float timestamp;
    public string pose;
    public string gesture;
    public string orientation;
    public float confidence;
    public float gesture_blend;
    public float gesture_confidence;
    public float orientation_value;
    public float orientation_confidence;
    // joints field is parsed separately
}

[System.Serializable]
public class PredictionData
{
    public string type;
    public float timestamp;
    public string pose;
    public float confidence;
    public JointsData joints;  // Changed from single position to multiple joints
}

[System.Serializable]
public class JointsData
{
    // Note: Unity's JsonUtility doesn't support Dictionary<int, JointData> directly
    // So we'll parse it manually or use a different approach
    // For now, store as Dictionary in code but handle parsing specially
}

[System.Serializable]
public class JointData
{
    public float timestamp;
    public int tag_id;
    public PositionVector tag_position;  // Original tag position
    public PositionVector joint_center;  // Calculated joint center (with offset)
    public float offset_applied;
    public RotationVector rotation;
    public RotationVector rotation_degrees;
}

[System.Serializable]
public class PositionVector
{
    public float x;
    public float y;
    public float z;
}

[System.Serializable]
public class RotationVector
{
    public float roll;
    public float pitch;
    public float yaw;
}

[System.Serializable]
public class AprilTagStatus
{
    public bool running;
    public int[] tracked_joints;
    // Note: latest_positions would need custom parsing
}
