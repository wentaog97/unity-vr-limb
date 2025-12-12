using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Controls multiple VR limb joints using multi-AprilTag tracking.
/// Each joint is identified by an AprilTag ID and has configurable offset distance.
/// 
/// Example Setup:
/// - Wrist: 4 tags with ID 1 (facing outward), offset = 0.03m
/// - Shoulder: 4 tags with ID 2 (facing outward), offset = 0.05m
/// 
/// Usage:
/// 1. Attach this script to a GameObject
/// 2. Configure joints in the Inspector (tag ID + offset + Transform)
/// 3. Start AprilTag tracking when connected
/// 4. Joints will automatically follow detected AprilTags
/// </summary>
public class MultiJointAprilTagController : MonoBehaviour
{
    [System.Serializable]
    public class JointConfig
    {
        [Tooltip("AprilTag ID for this joint (e.g., 1 for wrist, 2 for shoulder)")]
        public int tagId = 1;
        
        [Tooltip("Distance in meters from tag surface to joint center (positive = inward)")]
        public float offsetDistance = 0.03f;
        
        [Tooltip("The Transform to control (e.g., wrist joint)")]
        public Transform jointTransform;
        
        [Header("Position Mapping")]
        public Vector3 positionOffset = Vector3.zero;
        public float positionScale = 1.0f;
        
        [Header("Rotation Mapping")]
        public bool useRotation = true;
        public Vector3 rotationOffset = Vector3.zero;
        
        [Header("Smoothing")]
        [Range(0f, 0.9f)]
        public float positionSmoothing = 0.3f;
        [Range(0f, 0.9f)]
        public float rotationSmoothing = 0.3f;
        
        // Internal state
        [System.NonSerialized]
        public Vector3 smoothedPosition = Vector3.zero;
        [System.NonSerialized]
        public Quaternion smoothedRotation = Quaternion.identity;
        [System.NonSerialized]
        public bool hasInitialized = false;
    }
    
    [Header("References")]
    [Tooltip("ServerEMGInterpreter that handles tracking")]
    public ServerEMGInterpreter serverInterpreter;
    
    [Header("Joint Configuration")]
    [Tooltip("List of joints to track and control")]
    public List<JointConfig> joints = new List<JointConfig>();
    
    [Header("Auto-Setup")]
    [Tooltip("Automatically add joints to tracking when connected")]
    public bool autoAddJointsOnStart = true;
    
    [Tooltip("Automatically start AprilTag tracking when connected")]
    public bool autoStartTracking = true;
    
    [Header("Debug")]
    public bool showDebugInfo = false;
    public bool showGizmos = true;
    
    private bool isInitialized = false;
    
    private void Awake()
    {
        // Find ServerEMGInterpreter if not assigned
        if (!serverInterpreter)
        {
            serverInterpreter = FindFirstObjectByType<ServerEMGInterpreter>();
            if (!serverInterpreter)
            {
                Debug.LogError("[MultiJointController] ServerEMGInterpreter not found!");
            }
        }
    }
    
    private void Start()
    {
        if (autoAddJointsOnStart && autoStartTracking)
        {
            StartCoroutine(InitializeWhenConnected());
        }
    }
    
    private System.Collections.IEnumerator InitializeWhenConnected()
    {
        // Wait for server connection
        while (!serverInterpreter || !serverInterpreter.netClient || !serverInterpreter.netClient.IsConnected)
        {
            yield return new WaitForSeconds(0.5f);
        }
        
        yield return new WaitForSeconds(1.0f); // Give server time to initialize
        
        // Initialize tracking
        InitializeTracking();
    }
    
    /// <summary>
    /// Initialize AprilTag tracking for all configured joints
    /// </summary>
    [ContextMenu("Initialize Tracking")]
    public void InitializeTracking()
    {
        if (!serverInterpreter)
        {
            Debug.LogError("[MultiJointController] ServerEMGInterpreter not assigned!");
            return;
        }
        
        // Start AprilTag tracking on server
        if (autoStartTracking)
        {
            serverInterpreter.StartAprilTagTracking();
            Debug.Log("[MultiJointController] Started AprilTag tracking");
        }
        
        // Add all joints
        foreach (var joint in joints)
        {
            if (joint.jointTransform)
            {
                serverInterpreter.AddJoint(joint.tagId, joint.offsetDistance);
                Debug.Log($"[MultiJointController] Added joint: Tag ID {joint.tagId}, offset {joint.offsetDistance}m");
            }
            else
            {
                Debug.LogWarning($"[MultiJointController] Joint with tag ID {joint.tagId} has no Transform assigned!");
            }
        }
        
        isInitialized = true;
    }
    
    /// <summary>
    /// Stop tracking and clear all joints
    /// </summary>
    [ContextMenu("Stop Tracking")]
    public void StopTracking()
    {
        if (serverInterpreter)
        {
            serverInterpreter.ClearAllJoints();
            serverInterpreter.StopAprilTagTracking();
            Debug.Log("[MultiJointController] Stopped tracking");
        }
        
        isInitialized = false;
    }
    
    private void Update()
    {
        if (!isInitialized || !serverInterpreter) return;
        
        // Update each joint
        foreach (var joint in joints)
        {
            if (!joint.jointTransform) continue;
            
            // Get latest position for this joint
            var (jointCenter, rotation, hasData) = serverInterpreter.GetJointPosition(joint.tagId);
            
            if (!hasData)
            {
                if (showDebugInfo)
                {
                    Debug.Log($"[MultiJointController] No data for tag ID {joint.tagId}");
                }
                continue;
            }
            
            // First-time initialization
            if (!joint.hasInitialized)
            {
                joint.smoothedPosition = jointCenter;
                joint.smoothedRotation = Quaternion.Euler(rotation);
                joint.hasInitialized = true;
            }
            
            // Apply position mapping
            Vector3 mappedPosition = (jointCenter + joint.positionOffset) * joint.positionScale;
            
            // Apply smoothing
            joint.smoothedPosition = Vector3.Lerp(joint.smoothedPosition, mappedPosition, 1f - joint.positionSmoothing);
            
            // Apply position to joint
            joint.jointTransform.position = joint.smoothedPosition;
            
            // Apply rotation if enabled
            if (joint.useRotation)
            {
                Vector3 mappedRotation = rotation + joint.rotationOffset;
                Quaternion targetRotation = Quaternion.Euler(mappedRotation);
                joint.smoothedRotation = Quaternion.Slerp(joint.smoothedRotation, targetRotation, 1f - joint.rotationSmoothing);
                joint.jointTransform.rotation = joint.smoothedRotation;
            }
            
            if (showDebugInfo)
            {
                Debug.Log($"[MultiJointController] Tag {joint.tagId}: position={joint.smoothedPosition}, rotation={rotation}");
            }
        }
    }
    
    /// <summary>
    /// Add a new joint to track at runtime
    /// </summary>
    public void AddJoint(int tagId, float offsetDistance, Transform jointTransform)
    {
        var newJoint = new JointConfig
        {
            tagId = tagId,
            offsetDistance = offsetDistance,
            jointTransform = jointTransform
        };
        
        joints.Add(newJoint);
        
        if (isInitialized && serverInterpreter)
        {
            serverInterpreter.AddJoint(tagId, offsetDistance);
            Debug.Log($"[MultiJointController] Added joint at runtime: Tag ID {tagId}");
        }
    }
    
    /// <summary>
    /// Remove a joint from tracking
    /// </summary>
    public void RemoveJoint(int tagId)
    {
        joints.RemoveAll(j => j.tagId == tagId);
        
        if (isInitialized && serverInterpreter)
        {
            serverInterpreter.RemoveJoint(tagId);
            Debug.Log($"[MultiJointController] Removed joint: Tag ID {tagId}");
        }
    }
    
    /// <summary>
    /// Get the Transform for a specific joint by tag ID
    /// </summary>
    public Transform GetJointTransform(int tagId)
    {
        var joint = joints.Find(j => j.tagId == tagId);
        return joint?.jointTransform;
    }
    
    private void OnDrawGizmos()
    {
        if (!showGizmos || joints == null) return;
        
        // Draw each joint
        foreach (var joint in joints)
        {
            if (!joint.jointTransform) continue;
            
            // Draw sphere at joint position
            Gizmos.color = Color.green;
            Gizmos.DrawWireSphere(joint.jointTransform.position, 0.02f);
            
            // Draw label
            #if UNITY_EDITOR
            UnityEditor.Handles.Label(joint.jointTransform.position + Vector3.up * 0.05f, $"Tag {joint.tagId}");
            #endif
        }
    }
    
    private void OnDrawGizmosSelected()
    {
        if (!showGizmos || joints == null) return;
        
        // Draw coordinate axes for each joint
        foreach (var joint in joints)
        {
            if (!joint.jointTransform) continue;
            
            float axisLength = 0.05f;
            
            // X axis (red)
            Gizmos.color = Color.red;
            Gizmos.DrawRay(joint.jointTransform.position, joint.jointTransform.right * axisLength);
            
            // Y axis (green)
            Gizmos.color = Color.green;
            Gizmos.DrawRay(joint.jointTransform.position, joint.jointTransform.up * axisLength);
            
            // Z axis (blue)
            Gizmos.color = Color.blue;
            Gizmos.DrawRay(joint.jointTransform.position, joint.jointTransform.forward * axisLength);
        }
    }
}

