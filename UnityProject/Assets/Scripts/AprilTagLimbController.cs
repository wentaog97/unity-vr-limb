using UnityEngine;

/// <summary>
/// Controls VR limb joints using AprilTag position data from the server.
/// This script demonstrates how to use real-time AprilTag tracking to control limb movement.
/// 
/// Setup:
/// 1. Attach this script to a GameObject in your scene
/// 2. Assign the ServerEMGInterpreter reference (or it will auto-find)
/// 3. Assign the limb joint Transforms you want to control
/// 4. Adjust mapping settings (offset, scale, smoothing)
/// 5. Make sure server is running with AprilTag tracking enabled
/// </summary>
public class AprilTagLimbController : MonoBehaviour
{
    [Header("References")]
    [Tooltip("The ServerEMGInterpreter that receives position data")]
    public ServerEMGInterpreter serverInterpreter;
    
    [Header("AprilTag Configuration")]
    [Tooltip("AprilTag ID to track for this joint")]
    public int tagId = 1;
    
    [Tooltip("Offset distance from tag surface to joint center (meters)")]
    public float offsetDistance = 0.03f;
    
    [Header("Limb Joints to Control")]
    [Tooltip("Main joint to control (e.g., wrist or hand)")]
    public Transform targetJoint;
    
    [Tooltip("Optional: Parent joint (e.g., elbow) for IK calculations")]
    public Transform parentJoint;
    
    [Tooltip("Optional: Root joint (e.g., shoulder) for full arm IK")]
    public Transform rootJoint;
    
    [Header("Position Mapping")]
    [Tooltip("Offset to apply to AprilTag position (meters)")]
    public Vector3 positionOffset = Vector3.zero;
    
    [Tooltip("Scale factor for position (1.0 = 1:1 mapping)")]
    public float positionScale = 1.0f;
    
    [Tooltip("Coordinate system conversion (if needed)")]
    public bool invertX = false;
    public bool invertY = false;
    public bool invertZ = false;
    
    [Header("Rotation Mapping")]
    [Tooltip("Use AprilTag rotation to control joint rotation")]
    public bool useRotation = true;
    
    [Tooltip("Offset to apply to rotation (degrees)")]
    public Vector3 rotationOffset = Vector3.zero;

    [Tooltip("Tag's local axis that points toward the wrist (Unity space)")]
    public Vector3 tagForwardAxis = Vector3.forward;

    [Tooltip("Distance to move along tagForwardAxis to reach the wrist (meters)")]
    public float targetOffsetDistance = 0f;
    
    [Tooltip("Which axes to apply rotation to")]
    public bool applyRoll = true;
    public bool applyPitch = true;
    public bool applyYaw = true;
    
    [Header("Smoothing")]
    [Tooltip("Smooth position changes (0 = no smoothing, higher = more smoothing)")]
    [Range(0f, 0.9f)]
    public float positionSmoothing = 0.3f;
    
    [Tooltip("Smooth rotation changes")]
    [Range(0f, 0.9f)]
    public float rotationSmoothing = 0.3f;
    
    [Header("Constraints")]
    [Tooltip("Limit joint position to a defined region")]
    public bool usePositionConstraints = false;
    public Vector3 minPosition = new Vector3(-0.5f, -0.5f, -0.5f);
    public Vector3 maxPosition = new Vector3(0.5f, 0.5f, 0.5f);
    
    [Header("Calibration")]
    [Tooltip("Capture current AprilTag position as neutral/zero point")]
    public bool autoCalibrate = false;
    private Vector3 calibrationOffset = Vector3.zero;
    
    [Header("Debug")]
    public bool showDebugInfo = true;
    public bool showGizmos = true;
    
    // Internal state
    private Vector3 smoothedPosition = Vector3.zero;
    private Quaternion smoothedRotation = Quaternion.identity;
    private bool hasInitialPosition = false;
    private bool isCalibrated = false;
    
    private void Awake()
    {
        // Auto-find ServerEMGInterpreter if not assigned
        if (!serverInterpreter)
        {
            serverInterpreter = FindFirstObjectByType<ServerEMGInterpreter>();
            if (!serverInterpreter)
            {
                Debug.LogError("[AprilTagLimbController] ServerEMGInterpreter not found! Please assign it.");
            }
        }
        
        // Validate target joint
        if (!targetJoint)
        {
            Debug.LogError("[AprilTagLimbController] Target joint not assigned!");
        }
    }
    
    private     void Update()
    {
        if (!serverInterpreter || !targetJoint) return;
        
        // Get latest position data from server for this tag
        var (rawPosition, rawRotation, hasData) = serverInterpreter.GetJointPosition(tagId);
        
        if (!hasData)
        {
            if (showDebugInfo)
            {
                Debug.Log("[AprilTagLimbController] No AprilTag data available");
            }
            return;
        }
        
        // First-time initialization
        if (!hasInitialPosition)
        {
            smoothedPosition = rawPosition;
            smoothedRotation = Quaternion.Euler(rawRotation);
            hasInitialPosition = true;
            
            if (autoCalibrate)
            {
                CalibrateNeutralPosition();
            }
        }
        
        // Process position
        Vector3 processedPosition = ProcessPosition(rawPosition);
        
        // Process rotation
        Quaternion processedRotation = ProcessRotation(rawRotation);
        
        // Apply tag-based offset toward the wrist before smoothing
        Vector3 offsetDirection = GetTagForwardDirection(processedRotation);
        if (targetOffsetDistance != 0f)
        {
            processedPosition += offsetDirection * targetOffsetDistance;
        }

        // Apply smoothing
        smoothedPosition = Vector3.Lerp(smoothedPosition, processedPosition, 1f - positionSmoothing);
        smoothedRotation = Quaternion.Slerp(smoothedRotation, processedRotation, 1f - rotationSmoothing);
        
        // Apply to target joint
        ApplyToJoint(smoothedPosition, smoothedRotation);
        
        // Debug info
        if (showDebugInfo)
        {
            Debug.Log($"[AprilTagLimbController] Raw: {rawPosition}, Processed: {processedPosition}, Smoothed: {smoothedPosition}");
        }
    }
    
    private Vector3 ProcessPosition(Vector3 rawPosition)
    {
        Vector3 position = rawPosition;
        
        // Apply calibration offset
        if (isCalibrated)
        {
            position -= calibrationOffset;
        }
        
        // Apply coordinate system conversions
        position.x = invertX ? -position.x : position.x;
        position.y = invertY ? -position.y : position.y;
        position.z = invertZ ? -position.z : position.z;
        
        // Apply scale and offset
        position = position * positionScale + positionOffset;
        
        // Apply constraints
        if (usePositionConstraints)
        {
            position.x = Mathf.Clamp(position.x, minPosition.x, maxPosition.x);
            position.y = Mathf.Clamp(position.y, minPosition.y, maxPosition.y);
            position.z = Mathf.Clamp(position.z, minPosition.z, maxPosition.z);
        }
        
        return position;
    }
    
    private Quaternion ProcessRotation(Vector3 rawRotationDegrees)
    {
        if (!useRotation) return Quaternion.identity;
        
        // Select which axes to use
        Vector3 rotation = Vector3.zero;
        rotation.x = applyRoll ? rawRotationDegrees.x : 0f;
        rotation.y = applyPitch ? rawRotationDegrees.y : 0f;
        rotation.z = applyYaw ? rawRotationDegrees.z : 0f;
        
        // Apply offset
        rotation += rotationOffset;
        
        return Quaternion.Euler(rotation);
    }

    private Vector3 GetTagForwardDirection(Quaternion processedRotation)
    {
        Vector3 axis = tagForwardAxis == Vector3.zero ? Vector3.forward : tagForwardAxis.normalized;
        return processedRotation * axis;
    }
    
    private void ApplyToJoint(Vector3 position, Quaternion rotation)
    {
        // Apply position
        targetJoint.position = position;
        
        // Apply rotation
        if (useRotation)
        {
            targetJoint.rotation = rotation;
        }
        
        // If parent joints are assigned, you can implement IK here
        // Example: Simple 2-joint IK for elbow
        if (parentJoint && rootJoint)
        {
            // This is a placeholder - implement your IK solution
            // Unity has built-in IK or you can use FABRIK, CCD, etc.
            // SolveIK(rootJoint, parentJoint, targetJoint, position);
        }
    }
    
    /// <summary>
    /// Calibrate the current AprilTag position as the neutral/zero point
    /// </summary>
    [ContextMenu("Calibrate Neutral Position")]
    public void CalibrateNeutralPosition()
    {
        if (!serverInterpreter) return;
        
        var (position, rotation, hasData) = serverInterpreter.GetJointPosition(tagId);
        
        if (hasData)
        {
            calibrationOffset = position;
            isCalibrated = true;
            Debug.Log($"[AprilTagLimbController] Calibrated neutral position: {position}");
        }
        else
        {
            Debug.LogWarning("[AprilTagLimbController] Cannot calibrate - no AprilTag data");
        }
    }
    
    /// <summary>
    /// Reset calibration
    /// </summary>
    [ContextMenu("Reset Calibration")]
    public void ResetCalibration()
    {
        calibrationOffset = Vector3.zero;
        isCalibrated = false;
        Debug.Log("[AprilTagLimbController] Calibration reset");
    }
    
    /// <summary>
    /// Reset smoothing state
    /// </summary>
    [ContextMenu("Reset Smoothing")]
    public void ResetSmoothing()
    {
        hasInitialPosition = false;
        smoothedPosition = Vector3.zero;
        smoothedRotation = Quaternion.identity;
    }
    
    private void OnDrawGizmos()
    {
        if (!showGizmos || !targetJoint) return;
        
        // Draw current target joint position
        Gizmos.color = Color.green;
        Gizmos.DrawWireSphere(targetJoint.position, 0.02f);
        
        // Draw constraint bounds if enabled
        if (usePositionConstraints)
        {
            Gizmos.color = Color.yellow;
            Vector3 center = (minPosition + maxPosition) / 2f;
            Vector3 size = maxPosition - minPosition;
            Gizmos.DrawWireCube(center, size);
        }
        
        // Draw smoothed position
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireSphere(smoothedPosition, 0.015f);
        
        // Draw line from parent to target if available
        if (parentJoint)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawLine(parentJoint.position, targetJoint.position);
        }
        
        if (rootJoint && parentJoint)
        {
            Gizmos.DrawLine(rootJoint.position, parentJoint.position);
        }
    }
    
    private void OnDrawGizmosSelected()
    {
        if (!showGizmos || !targetJoint) return;
        
        // Draw coordinate axes at target joint
        float axisLength = 0.1f;
        
        // X axis (red)
        Gizmos.color = Color.red;
        Gizmos.DrawRay(targetJoint.position, targetJoint.right * axisLength);
        
        // Y axis (green)
        Gizmos.color = Color.green;
        Gizmos.DrawRay(targetJoint.position, targetJoint.up * axisLength);
        
        // Z axis (blue)
        Gizmos.color = Color.blue;
        Gizmos.DrawRay(targetJoint.position, targetJoint.forward * axisLength);
    }
}

