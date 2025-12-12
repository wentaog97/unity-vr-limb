// WristOffsetsController.cs
// Adds natural wrist rotation on top of an IK-driven arm:
// - Hand inherits forearm (FK feel) via RotationConstraint
// - We add model-driven flex/extend and sup/pronation around stable axes from WristSpace

using UnityEngine;

public class WristOffsetsController : MonoBehaviour
{
    [Header("Rig References")]
    [Tooltip("Forearm (lower arm) bone Transform.")]
    public Transform forearm;
    [Tooltip("Hand/Wrist bone Transform (the tip of the Two-Bone chain).")]
    public Transform hand;
    [Tooltip("Empty GameObject under Forearm that defines wrist axes: +Z along forearm, +X = flex/extend.")]
    public Transform wristSpace;

    [Header("Inputs (set these every frame from your models)")]
    [Tooltip("Flex(+)/Extend(-). Degrees unless inputIsRadians is true.")]
    public float flexInput = 0f;
    [Tooltip("Supinate(+)/Pronate(-). Degrees unless inputIsRadians is true.")]
    public float supInput  = 0f;
    [Tooltip("If true, flexInput / supInput are in radians; they will be converted to degrees.")]
    public bool inputIsRadians = false;

    [Header("Input Overrides")]
    [Tooltip("When false, ignore incoming supination values and use manualSupDegrees instead.")]
    public bool useModelSupInput = true;
    [Tooltip("Manual supination angle (degrees) applied when useModelSupInput = false.")]
    public float manualSupDegrees = 0f;

    [Header("Tuning")]
    [Tooltip("Left arm usually needs sign flips for axes. Check these if your rotations feel backwards.")]
    public bool isLeftArm = false;
    [Tooltip("Scale the incoming angles (e.g., 1.0 = use as-is, 0.5 = halve).")]
    public float flexGain = 1.0f;
    public float supGain  = 1.0f;

    [Tooltip("Clamp limits for flex/extend (deg). Typical: min=-70, max=80")]
    public Vector2 flexClampDeg = new Vector2(-70f, 80f);
    [Tooltip("Clamp limits for supination/pronation (deg). Typical: min=-120, max=240")]
    public Vector2 supClampDeg  = new Vector2(-120f, 240f);

    [Tooltip("How much to apply (0 = ignore offsets, 1 = full effect).")]
    [Range(0f, 1f)] public float weight = 1.0f;

    [Header("Smoothing")]
    [Tooltip("Time (seconds) to smooth flex/extend. 0 = no smoothing.")]
    public float flexSmoothTime = 0.06f;
    [Tooltip("Time (seconds) to smooth sup/pronation. 0 = no smoothing.")]
    public float supSmoothTime  = 0.06f;

    [Header("Optional: distribute twist along forearm")]
    [Tooltip("Apply a fraction of supination/pronation as twist on the forearm for more anatomical look.")]
    [Range(0f, 1f)] public float forearmTwistShare = 0.5f;

    // Internal state
    float _flexDegSmoothed, _supDegSmoothed;
    float _flexVel, _supVel;

    void Reset()
    {
        weight = 1f;
        flexClampDeg = new Vector2(-70f, 80f);
        supClampDeg  = new Vector2(-120f, 240f);
        flexSmoothTime = 0.06f;
        supSmoothTime  = 0.06f;
        forearmTwistShare = 0.5f;
    }

    void LateUpdate()
    {
        if (forearm == null || hand == null || wristSpace == null) return;

        // 1) Convert inputs to degrees and apply gains
        float flexDeg = inputIsRadians ? flexInput * Mathf.Rad2Deg : flexInput;  // + = flex, - = extend
        float supDegInput = inputIsRadians ? supInput * Mathf.Rad2Deg : supInput;
        float supDeg  = useModelSupInput ? supDegInput : manualSupDegrees;   // + = supinate, - = pronate
        flexDeg *= flexGain;
        supDeg  *= supGain;

        // 2) Arm-side conventions
        // With X as the twist axis, the sign for supination often flips on the left arm.
        if (isLeftArm)
        {
            supDeg = -supDeg;    // comment this out if your rig already mirrors twist correctly
            // If flex feels backwards on your rig, uncomment the next line:
            // flexDeg = -flexDeg;
        }

        // 3) Clamp
        flexDeg = Mathf.Clamp(flexDeg, flexClampDeg.x, flexClampDeg.y);
        supDeg  = Mathf.Clamp(supDeg,  supClampDeg.x,  supClampDeg.y);

        // 4) Smooth (angle-aware)
        if (flexSmoothTime > 0f)
            _flexDegSmoothed = Mathf.SmoothDampAngle(_flexDegSmoothed, flexDeg, ref _flexVel, flexSmoothTime);
        else
            _flexDegSmoothed = flexDeg;

        if (supSmoothTime > 0f)
            _supDegSmoothed = Mathf.SmoothDamp(_supDegSmoothed, supDeg, ref _supVel, supSmoothTime);
        else
            _supDegSmoothed = supDeg;

        // 5) Build additive rotations in the NEW wristSpace basis
        //    - Twist around +X (forearm axis)
        //    - Flex/Extend around +Z (left-right hinge across the wrist)
        Quaternion baseRot = hand.rotation; // already following forearm via RotationConstraint
        Vector3 twistAxis  = wristSpace.right;    // X
        Vector3 flexAxis   = wristSpace.up;  // Z

        Quaternion twistQ = Quaternion.AngleAxis(_supDegSmoothed, twistAxis);
        Quaternion flexQ  = Quaternion.AngleAxis(_flexDegSmoothed, flexAxis);

        // Order: apply twist, then flex
        Quaternion targetHandRot = (twistQ * flexQ) * baseRot;

        // 6) Optional: share part of the twist with the forearm so it doesn't all happen at the wrist
        if (forearmTwistShare > 0f)
        {
            float forearmTwistDeg = _supDegSmoothed * forearmTwistShare;
            float wristTwistDeg   = _supDegSmoothed * (1f - forearmTwistShare);

            Quaternion forearmTwistQ = Quaternion.AngleAxis(forearmTwistDeg, twistAxis);
            Quaternion wristTwistQ   = Quaternion.AngleAxis(wristTwistDeg,  twistAxis);
            Quaternion flexOnlyQ     = Quaternion.AngleAxis(_flexDegSmoothed, flexAxis);

            // Apply forearm twist in world space around the same axis
            forearm.rotation = forearmTwistQ * forearm.rotation;

            // Recompute hand target so forearm+wrist twist sum to the original
            targetHandRot = (wristTwistQ * flexOnlyQ) * baseRot;
        }

        // 7) Blend and apply
        hand.rotation = (weight < 1f)
            ? Quaternion.Slerp(baseRot, targetHandRot, Mathf.Clamp01(weight))
            : targetHandRot;
    }


    // Convenience setters if you want to push values from another script
    public void SetFlexDegrees(float deg) => flexInput = inputIsRadians ? deg * Mathf.Deg2Rad : deg;
    public void SetSupDegrees(float deg)  => supInput  = inputIsRadians ? deg * Mathf.Deg2Rad : deg;
}
