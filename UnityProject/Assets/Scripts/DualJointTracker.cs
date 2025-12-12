using UnityEngine;
using TMPro;

/// <summary>
/// Tracks 2 AprilTag joints and displays their positions.
/// 
/// SETUP:
/// 1. Attach to any GameObject
/// 2. Drag TextMeshPro text boxes into Inspector slots
/// 3. Set tag IDs and offsets in Inspector
/// 4. Run - it will auto-start tracking when connected
/// </summary>
public class DualJointTracker : MonoBehaviour
{
    [Header("Tag Configuration")]
    [Tooltip("First joint's AprilTag ID (e.g., 1 for wrist)")]
    public int joint1TagId = 1;
    
    [Tooltip("Offset distance for joint 1 in meters (e.g., 0.03 = 3cm)")]
    public float joint1Offset = 0.03f;
    
    [Tooltip("Second joint's AprilTag ID (e.g., 2 for shoulder)")]
    public int joint2TagId = 2;
    
    [Tooltip("Offset distance for joint 2 in meters (e.g., 0.05 = 5cm)")]
    public float joint2Offset = 0.05f;

    [Header("Tracker Origin Offset")]
    [Tooltip("Optional global position offset applied to all incoming AprilTag data (meters)")]
    public Vector3 trackerPositionOffset = Vector3.zero;
    
    [Header("Display Text Boxes (drag TextMeshPro here)")]
    public TextMeshProUGUI statusText;
    public TextMeshProUGUI tag1OutputText;
    public TextMeshProUGUI tag2OutputText;
    public TextMeshProUGUI debugText;
    
    [Header("Optional: Control These Transforms")]
    [Tooltip("Transform to control for joint 1 (e.g., wrist IK target)")]
    public Transform joint1Transform;
    
    [Tooltip("Apply position from joint 1 tracking data")]
    public bool joint1ApplyPosition = true;
    
    [Tooltip("Apply rotation from joint 1 tracking data")]
    public bool joint1ApplyRotation = false;
    
    [Space(10)]
    [Tooltip("Transform to control for joint 2 (e.g., shoulder/root)")]
    public Transform joint2Transform;
    
    [Tooltip("Apply position from joint 2 tracking data")]
    public bool joint2ApplyPosition = true;
    
    [Tooltip("Apply rotation from joint 2 tracking data")]
    public bool joint2ApplyRotation = false;
    
    [Header("Positioning Mode")]
    [Tooltip("Absolute: Directly set position from tracker | Relative: Track movement from calibrated reference")]
    public bool useRelativePositioning = false;
    
    [Tooltip("Auto-calibrate reference positions when tracking starts")]
    public bool autoCalibrate = true;
    
    [Header("Sensitivity Adjustments")]
    [Tooltip("Multiplier for position changes (1.0 = normal, 0.5 = half speed, 2.0 = double speed)")]
    [Range(0.1f, 3.0f)]
    public float positionSensitivity = 1.0f;
    
    [Tooltip("Multiplier for rotation changes (only if rotation is applied)")]
    [Range(0.1f, 3.0f)]
    public float rotationSensitivity = 1.0f;
    
    [Header("Axis Inversion (Fix Coordinate System Mismatches)")]
    [Tooltip("Invert X axis (left/right)")]
    public bool invertX = false;
    
    [Tooltip("Invert Y axis (up/down) - Enable if up/down movement is reversed")]
    public bool invertY = true;
    
    [Tooltip("Invert Z axis (forward/back)")]
    public bool invertZ = false;
    
    [Header("Calibration (Relative Mode Only)")]
    [Tooltip("To recalibrate: Right-click script in Inspector → 'Calibrate Reference', or call CalibrateReference() from UI button")]
    public bool showCalibrationInfo = true;
    
    // Internal references (auto-found)
    private ServerEMGInterpreter serverInterpreter;
    private NetworkBridgeClient networkClient;
    
    private bool isInitialized = false;
    private bool isConnected = false;
    private bool jointsAdded = false;  // Track if joints have been added
    private float lastPollTime = 0f;
    private float initTime = 0f;  // Track when initialization happened
    private const float pollInterval = 0.1f;  // Poll 10 times per second
    private const float jointAddDelay = 1.0f;  // Delay after init before adding joints
    
    // Calibration data for relative positioning
    private bool isCalibrated = false;
    private Vector3 joint1ReferenceTracked;  // Where the tracker was when we calibrated
    private Vector3 joint1ReferenceTransform;  // Where the transform was when we calibrated
    private Quaternion joint1ReferenceRotTracked;  // Tracker rotation at calibration
    private Quaternion joint1ReferenceRotTransform;  // Transform rotation at calibration
    private Vector3 joint2ReferenceTracked;
    private Vector3 joint2ReferenceTransform;
    private Quaternion joint2ReferenceRotTracked;
    private Quaternion joint2ReferenceRotTransform;
    
    void Start()
    {
        Debug.Log("[DualJointTracker] Start() called");
        
        // Find components automatically
        serverInterpreter = FindFirstObjectByType<ServerEMGInterpreter>();
        networkClient = FindFirstObjectByType<NetworkBridgeClient>();
        
        if (!serverInterpreter)
        {
            Debug.LogError("[DualJointTracker] ServerEMGInterpreter NOT FOUND!");
            UpdateStatus("ERROR: ServerEMGInterpreter not found in scene!");
            return;
        }
        else
        {
            Debug.Log("[DualJointTracker] ServerEMGInterpreter found: " + serverInterpreter.name);
        }
        
        if (!networkClient)
        {
            Debug.LogError("[DualJointTracker] NetworkBridgeClient NOT FOUND!");
            UpdateStatus("ERROR: NetworkBridgeClient not found in scene!");
            return;
        }
        else
        {
            Debug.Log("[DualJointTracker] NetworkBridgeClient found: " + networkClient.name);
        }
        
        Debug.Log($"[DualJointTracker] Configuration: Joint1_TagID={joint1TagId}, Offset={joint1Offset}, Joint2_TagID={joint2TagId}, Offset={joint2Offset}");
        UpdateStatus("Ready. Waiting for server connection...");
    }
    
    void Update()
    {
        // Check connection status
        bool currentlyConnected = networkClient && networkClient.IsConnected;
        
        if (currentlyConnected && !isConnected)
        {
            // Just connected
            OnConnected();
        }
        else if (!currentlyConnected && isConnected)
        {
            // Just disconnected
            OnDisconnected();
        }
        
        isConnected = currentlyConnected;
        
        // Check if we need to add joints (safety check in case Invoke didn't work)
        if (isInitialized && !jointsAdded && Time.time - initTime > jointAddDelay)
        {
            Debug.LogWarning("[DualJointTracker] Safety check: Adding joints manually (Invoke may have failed)");
            AddJointsDelayed();
        }
        
        // Note: Keyboard calibration removed due to Input System compatibility
        // Use Context Menu (right-click in Inspector → "Calibrate Reference")
        // Or call CalibrateReference() from a UI button
        
        // Auto-calibrate when first data arrives (in relative mode)
        if (useRelativePositioning && autoCalibrate && !isCalibrated && isInitialized && isConnected)
        {
            // Check if we have valid tracking data
            var (_, _, has1) = serverInterpreter.GetJointPosition(joint1TagId);
            var (_, _, has2) = serverInterpreter.GetJointPosition(joint2TagId);
            
            if ((has1 && joint1Transform) || (has2 && joint2Transform))
            {
                CalibrateReference();
            }
        }
        
        // Update displays if initialized
        if (isInitialized && isConnected)
        {
            // Poll for joint positions periodically
            if (Time.time - lastPollTime > pollInterval)
            {
                RequestJointPositions();
                lastPollTime = Time.time;
            }
            
            UpdateJointDisplays();
        }
    }
    
    void RequestJointPositions()
    {
        if (networkClient && networkClient.IsConnected)
        {
            networkClient.QueueMessage("GET_JOINT_POSITIONS");
        }
    }
    
    void OnConnected()
    {
        UpdateStatus("Connected! Starting AprilTag tracking...");
        
        // Wait a moment for server to be ready, then initialize
        Invoke(nameof(InitializeTracking), 1.0f);
    }
    
    void OnDisconnected()
    {
        isInitialized = false;
        jointsAdded = false;  // Reset joint tracking
        isCalibrated = false;  // Reset calibration on disconnect
        UpdateStatus("Disconnected from server");
        
        if (tag1OutputText) tag1OutputText.text = "Tag 1: No data";
        if (tag2OutputText) tag2OutputText.text = "Tag 2: No data";
    }
    
    void InitializeTracking()
    {
        Debug.Log("[DualJointTracker] InitializeTracking called");
        
        if (!serverInterpreter)
        {
            Debug.LogError("[DualJointTracker] serverInterpreter is null, cannot initialize!");
            return;
        }
        
        if (!networkClient.IsConnected)
        {
            Debug.LogWarning("[DualJointTracker] Network not connected, cannot initialize!");
            return;
        }
        
        // Prevent duplicate initialization
        if (isInitialized)
        {
            Debug.Log("[DualJointTracker] Already initialized, skipping...");
            return;
        }
        
        UpdateStatus("Starting camera...");
        
        // Start AprilTag tracking
        Debug.Log("[DualJointTracker] Starting AprilTag tracking...");
        serverInterpreter.StartAprilTagTracking();
        
        // Mark as initialized to prevent duplicate calls
        isInitialized = true;
        initTime = Time.time;  // Record initialization time
        
        // Add both joints after small delay for camera to init
        Debug.Log($"[DualJointTracker] Scheduling AddJointsDelayed in {jointAddDelay}s");
        Invoke(nameof(AddJointsDelayed), jointAddDelay);
        
        Debug.Log($"[DualJointTracker] Initialization complete, waiting for joints to be added");
    }
    
    void AddJointsDelayed()
    {
        Debug.Log("[DualJointTracker] AddJointsDelayed called");
        
        if (!serverInterpreter)
        {
            Debug.LogError("[DualJointTracker] serverInterpreter is null!");
            return;
        }
        
        // Prevent duplicate joint additions
        if (jointsAdded)
        {
            Debug.Log("[DualJointTracker] Joints already added, skipping duplicate call");
            return;
        }
        
        Debug.Log($"[DualJointTracker] Adding joint 1 (TagID={joint1TagId}) with offset {joint1Offset}m");
        serverInterpreter.AddJoint(joint1TagId, joint1Offset);
        
        Debug.Log($"[DualJointTracker] Adding joint 2 (TagID={joint2TagId}) with offset {joint2Offset}m");
        serverInterpreter.AddJoint(joint2TagId, joint2Offset);
        
        jointsAdded = true;  // Mark as added
        
        UpdateStatus($"Tracking TagID {joint1TagId} and TagID {joint2TagId}");
        UpdateDebug($"Joint1 TagID={joint1TagId}: offset={joint1Offset}m\nJoint2 TagID={joint2TagId}: offset={joint2Offset}m\n\nMove tags in front of camera...");
        
        Debug.Log($"[DualJointTracker] Joints added successfully");
    }
    
    void UpdateJointDisplays()
    {
        // Get joint 1 position (using its tag ID)
        var (pos1, rot1, has1) = serverInterpreter.GetJointPosition(joint1TagId);
        if (has1)
        {
            Vector3 offsetPos1 = ApplyGlobalPositionOffset(pos1);
            Vector3 adjustedPos1 = ApplyAxisInversion(offsetPos1);
            
            // Calculate final position and rotation (absolute or relative)
            Vector3 finalPos1 = adjustedPos1;
            Quaternion finalRot1 = Quaternion.Euler(rot1);
            
            if (useRelativePositioning && isCalibrated)
            {
                // Relative mode: current = reference transform + sensitivity * (current tracked - reference tracked)
                Vector3 trackedDelta = adjustedPos1 - joint1ReferenceTracked;
                finalPos1 = joint1ReferenceTransform + (trackedDelta * positionSensitivity);
                
                // Rotation: apply relative rotation with sensitivity
                Quaternion trackedRotDelta = Quaternion.Euler(rot1) * Quaternion.Inverse(joint1ReferenceRotTracked);
                finalRot1 = joint1ReferenceRotTransform * Quaternion.Slerp(Quaternion.identity, trackedRotDelta, rotationSensitivity);
            }
            else if (!useRelativePositioning)
            {
                // Absolute mode: apply sensitivity as a scale from origin
                finalPos1 = adjustedPos1 * positionSensitivity;
                // Rotation sensitivity in absolute mode (partial rotation application)
                if (rotationSensitivity != 1.0f)
                {
                    finalRot1 = Quaternion.Slerp(Quaternion.identity, Quaternion.Euler(rot1), rotationSensitivity);
                }
            }
            
            // Display
            string output1 = $"Joint 1 (TagID {joint1TagId}) DETECTED:\n";
            output1 += $"Raw Pos: ({pos1.x:F3}, {pos1.y:F3}, {pos1.z:F3})\n";
            if (useRelativePositioning)
            {
                output1 += $"Final Pos: ({finalPos1.x:F3}, {finalPos1.y:F3}, {finalPos1.z:F3})\n";
                output1 += isCalibrated ? "[Calibrated]" : "[Not Calibrated - See Inspector]";
            }
            output1 += $"\nRotation: ({rot1.x:F1}°, {rot1.y:F1}°, {rot1.z:F1}°)\n";
            output1 += $"Offset: {joint1Offset}m | Sens: {positionSensitivity:F1}x";
            
            if (tag1OutputText) tag1OutputText.text = output1;
            
            // Update transform if assigned
            if (joint1Transform)
            {
                if (joint1ApplyPosition)
                {
                    joint1Transform.position = finalPos1;
                }
                
                if (joint1ApplyRotation)
                {
                    joint1Transform.rotation = finalRot1;
                }
            }
        }
        else
        {
            if (tag1OutputText) tag1OutputText.text = $"Joint 1 (TagID {joint1TagId}): Not visible";
        }
        
        // Get joint 2 position (using its tag ID)
        var (pos2, rot2, has2) = serverInterpreter.GetJointPosition(joint2TagId);
        if (has2)
        {
            Vector3 offsetPos2 = ApplyGlobalPositionOffset(pos2);
            Vector3 adjustedPos2 = ApplyAxisInversion(offsetPos2);
            
            // Calculate final position and rotation (absolute or relative)
            Vector3 finalPos2 = adjustedPos2;
            Quaternion finalRot2 = Quaternion.Euler(rot2);
            
            if (useRelativePositioning && isCalibrated)
            {
                // Relative mode: current = reference transform + sensitivity * (current tracked - reference tracked)
                Vector3 trackedDelta = adjustedPos2 - joint2ReferenceTracked;
                finalPos2 = joint2ReferenceTransform + (trackedDelta * positionSensitivity);
                
                // Rotation: apply relative rotation with sensitivity
                Quaternion trackedRotDelta = Quaternion.Euler(rot2) * Quaternion.Inverse(joint2ReferenceRotTracked);
                finalRot2 = joint2ReferenceRotTransform * Quaternion.Slerp(Quaternion.identity, trackedRotDelta, rotationSensitivity);
            }
            else if (!useRelativePositioning)
            {
                // Absolute mode: apply sensitivity as a scale from origin
                finalPos2 = adjustedPos2 * positionSensitivity;
                // Rotation sensitivity in absolute mode (partial rotation application)
                if (rotationSensitivity != 1.0f)
                {
                    finalRot2 = Quaternion.Slerp(Quaternion.identity, Quaternion.Euler(rot2), rotationSensitivity);
                }
            }
            
            // Display
            string output2 = $"Joint 2 (TagID {joint2TagId}) DETECTED:\n";
            output2 += $"Raw Pos: ({pos2.x:F3}, {pos2.y:F3}, {pos2.z:F3})\n";
            if (useRelativePositioning)
            {
                output2 += $"Final Pos: ({finalPos2.x:F3}, {finalPos2.y:F3}, {finalPos2.z:F3})\n";
                output2 += isCalibrated ? "[Calibrated]" : "[Not Calibrated - See Inspector]";
            }
            output2 += $"\nRotation: ({rot2.x:F1}°, {rot2.y:F1}°, {rot2.z:F1}°)\n";
            output2 += $"Offset: {joint2Offset}m | Sens: {positionSensitivity:F1}x";
            
            if (tag2OutputText) tag2OutputText.text = output2;
            
            // Update transform if assigned
            if (joint2Transform)
            {
                if (joint2ApplyPosition)
                {
                    joint2Transform.position = finalPos2;
                }
                
                if (joint2ApplyRotation)
                {
                    joint2Transform.rotation = finalRot2;
                }
            }
        }
        else
        {
            if (tag2OutputText) tag2OutputText.text = $"Joint 2 (TagID {joint2TagId}): Not visible";
        }
        
        // Update debug info
        int detectedCount = (has1 ? 1 : 0) + (has2 ? 1 : 0);
        string mode = useRelativePositioning ? "Relative" : "Absolute";
        string calibStatus = useRelativePositioning ? (isCalibrated ? " [CAL]" : " [UNCAL]") : "";
        UpdateDebug($"Mode: {mode}{calibStatus}\nDetected: {detectedCount}/2 tags\nJoint1 (TagID {joint1TagId}): {(has1 ? "✓" : "✗")}\nJoint2 (TagID {joint2TagId}): {(has2 ? "✓" : "✗")}");
    }
    
    /// <summary>
    /// Calibrate reference positions for relative tracking mode.
    /// Captures current arm positions and tracker positions as the "zero point".
    /// After calibration, tracker movements are applied as deltas from this reference.
    /// </summary>
    [ContextMenu("Calibrate Reference")]
    public void CalibrateReference()
    {
        bool calibrated = false;
        
        // Calibrate joint 1 if we have tracking data and a transform
        var (pos1, rot1, has1) = serverInterpreter.GetJointPosition(joint1TagId);
        if (has1 && joint1Transform)
        {
            Vector3 offsetPos1 = ApplyAxisInversion(ApplyGlobalPositionOffset(pos1));
            joint1ReferenceTracked = offsetPos1;
            joint1ReferenceTransform = joint1Transform.position;
            joint1ReferenceRotTracked = Quaternion.Euler(rot1);
            joint1ReferenceRotTransform = joint1Transform.rotation;
            calibrated = true;
            Debug.Log($"[DualJointTracker] Calibrated Joint 1: Transform at {joint1ReferenceTransform}, Tracker at {joint1ReferenceTracked}");
        }
        
        // Calibrate joint 2 if we have tracking data and a transform
        var (pos2, rot2, has2) = serverInterpreter.GetJointPosition(joint2TagId);
        if (has2 && joint2Transform)
        {
            Vector3 offsetPos2 = ApplyAxisInversion(ApplyGlobalPositionOffset(pos2));
            joint2ReferenceTracked = offsetPos2;
            joint2ReferenceTransform = joint2Transform.position;
            joint2ReferenceRotTracked = Quaternion.Euler(rot2);
            joint2ReferenceRotTransform = joint2Transform.rotation;
            calibrated = true;
            Debug.Log($"[DualJointTracker] Calibrated Joint 2: Transform at {joint2ReferenceTransform}, Tracker at {joint2ReferenceTracked}");
        }
        
        if (calibrated)
        {
            isCalibrated = true;
            UpdateStatus("Calibrated! (Right-click script → Calibrate to recalibrate)");
            Debug.Log("[DualJointTracker] Calibration complete!");
        }
        else
        {
            Debug.LogWarning("[DualJointTracker] Cannot calibrate - no tracking data or transforms assigned");
        }
    }
    
    private Vector3 ApplyGlobalPositionOffset(Vector3 rawPosition)
    {
        return rawPosition + trackerPositionOffset;
    }

    private Vector3 ApplyAxisInversion(Vector3 position)
    {
        return new Vector3(
            invertX ? -position.x : position.x,
            invertY ? -position.y : position.y,
            invertZ ? -position.z : position.z
        );
    }

    void UpdateStatus(string message)
    {
        if (statusText) statusText.text = message;
        Debug.Log($"[DualJointTracker] {message}");
    }
    
    void UpdateDebug(string message)
    {
        if (debugText) debugText.text = message;
    }
    
    // Public methods you can call from buttons
    [ContextMenu("Restart Tracking")]
    public void RestartTracking()
    {
        if (isConnected)
        {
            isInitialized = false;
            jointsAdded = false;  // Reset joint tracking
            isCalibrated = false;  // Reset calibration
            serverInterpreter.ClearAllJoints();
            Invoke(nameof(InitializeTracking), 0.5f);
        }
    }
    
    [ContextMenu("Stop Tracking")]
    public void StopTracking()
    {
        if (serverInterpreter)
        {
            serverInterpreter.ClearAllJoints();
            serverInterpreter.StopAprilTagTracking();
            isInitialized = false;
            UpdateStatus("Tracking stopped");
        }
    }
}

