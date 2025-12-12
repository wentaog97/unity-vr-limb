// AnkleOffsetsController.cs
// Mirrors WristOffsetsController but drives ankle/calf instead of wrist/forearm.
// - Foot inherits calf pose via rig. This script layers dorsiflexion/plantarflexion and calf rotation
//   using a stable local space (ankleSpace) so EMG inputs remain anatomically consistent.

using UnityEngine;

public class AnkleOffsetsController : MonoBehaviour
{
    [Header("Rig References")]
    [Tooltip("Calf (lower leg) bone Transform.")]
    public Transform calf;
    [Tooltip("Foot/Ankle bone Transform (end of the leg chain).")]
    public Transform foot;
    [Tooltip("Empty GameObject under Calf that defines ankle axes: +X along calf, +Y = dorsiflex/plantar.")]
    public Transform ankleSpace;
    [Tooltip("Local axis (relative to ankleSpace) that represents dorsiflex/plantar hinge. Default = +X (left-right).")]
    public Vector3 flexAxisLocal = Vector3.right;
    [Tooltip("Local axis (relative to ankleSpace) for calf rotation/internal-external twist. Default = +Y (along calf).")]
    public Vector3 calfAxisLocal = Vector3.up;

    [Header("Inputs (set every frame)")]
    [Tooltip("Dorsiflex(+)/Plantarflex(-). Degrees unless inputIsRadians is true.")]
    public float flexInput = 0f;
    [Tooltip("External(+)/Internal(-) calf rotation. Degrees unless inputIsRadians is true.")]
    public float calfRotInput = 0f;
    [Tooltip("If true, flexInput / calfRotInput are provided in radians.")]
    public bool inputIsRadians = false;

    [Header("Tuning")]
    [Tooltip("Flip signs for left leg rigs if rotations feel backwards.")]
    public bool isLeftLeg = false;
    [Tooltip("Scale the incoming dorsiflex/plantar angle (1 = raw input).")]
    public float flexGain = 1f;
    [Tooltip("Scale the incoming calf rotation angle (1 = raw input).")]
    public float calfRotGain = 1f;

    [Tooltip("Clamp limits for dorsiflex/plantarflex (deg). Typical: min=-65, max=35")]
    public Vector2 flexClampDeg = new Vector2(-65f, 35f);
    [Tooltip("Clamp limits for calf rotation (deg). Typical: min=-45, max=45")]
    public Vector2 calfRotClampDeg = new Vector2(-45f, 45f);

    [Tooltip("How much to apply (0 = ignore offsets, 1 = full effect).")]
    [Range(0f, 1f)] public float weight = 1f;

    [Header("Smoothing")]
    [Tooltip("Time (seconds) to smooth dorsiflex/plantar.")]
    public float flexSmoothTime = 0.08f;
    [Tooltip("Time (seconds) to smooth calf rotation.")]
    public float calfRotSmoothTime = 0.08f;

    [Header("Optional: distribute twist along calf")]
    [Tooltip("Apply a fraction of calf rotation directly on the calf bone so the foot is not solely responsible.")]
    [Range(0f, 1f)] public float calfTwistShare = 0.35f;

    // Internal state
    float _flexDegSmoothed, _calfRotDegSmoothed;
    float _flexVel, _calfRotVel;

    void Reset()
    {
        weight = 1f;
        flexClampDeg = new Vector2(-65f, 35f);
        calfRotClampDeg = new Vector2(-45f, 45f);
        flexSmoothTime = 0.08f;
        calfRotSmoothTime = 0.08f;
        calfTwistShare = 0.35f;
    }

    void LateUpdate()
    {
        if (calf == null || foot == null || ankleSpace == null) return;

        // 1) Convert inputs to degrees and apply gains
        float flexDeg = inputIsRadians ? flexInput * Mathf.Rad2Deg : flexInput;      // + = dorsiflex, - = plantarflex
        float calfRotDeg = inputIsRadians ? calfRotInput * Mathf.Rad2Deg : calfRotInput; // + = external, - = internal
        flexDeg *= flexGain;
        calfRotDeg *= calfRotGain;

        // 2) Left/right leg conventions
        if (isLeftLeg)
        {
            calfRotDeg = -calfRotDeg;
            // Uncomment if dorsiflex feels backwards for your rig:
            // flexDeg = -flexDeg;
        }

        // 3) Clamp
        flexDeg = Mathf.Clamp(flexDeg, flexClampDeg.x, flexClampDeg.y);
        calfRotDeg = Mathf.Clamp(calfRotDeg, calfRotClampDeg.x, calfRotClampDeg.y);

        // 4) Smooth
        if (flexSmoothTime > 0f)
            _flexDegSmoothed = Mathf.SmoothDampAngle(_flexDegSmoothed, flexDeg, ref _flexVel, flexSmoothTime);
        else
            _flexDegSmoothed = flexDeg;

        if (calfRotSmoothTime > 0f)
            _calfRotDegSmoothed = Mathf.SmoothDampAngle(_calfRotDegSmoothed, calfRotDeg, ref _calfRotVel, calfRotSmoothTime);
        else
            _calfRotDegSmoothed = calfRotDeg;

        // 5) Build additive rotations in ankleSpace basis
        Quaternion baseRot = foot.rotation;
        Vector3 twistAxis = GetAxisWorld(ankleSpace, calfAxisLocal, Vector3.up);
        Vector3 flexAxis  = GetAxisWorld(ankleSpace, flexAxisLocal, Vector3.right);

        Quaternion twistQ = Quaternion.AngleAxis(_calfRotDegSmoothed, twistAxis);
        Quaternion flexQ = Quaternion.AngleAxis(_flexDegSmoothed, flexAxis);

        Quaternion targetFootRot = (twistQ * flexQ) * baseRot;

        // 6) Share calf rotation if requested so foot is not forced to align fully with calf
        if (calfTwistShare > 0f)
        {
            float calfShareDeg = _calfRotDegSmoothed * calfTwistShare;
            float footShareDeg = _calfRotDegSmoothed * (1f - calfTwistShare);

            Quaternion calfShareQ = Quaternion.AngleAxis(calfShareDeg, twistAxis);
            Quaternion footShareQ = Quaternion.AngleAxis(footShareDeg, twistAxis);
            Quaternion flexOnlyQ = Quaternion.AngleAxis(_flexDegSmoothed, flexAxis);

            calf.rotation = calfShareQ * calf.rotation;
            targetFootRot = (footShareQ * flexOnlyQ) * baseRot;
        }

        // 7) Blend and apply
        foot.rotation = (weight < 1f)
            ? Quaternion.Slerp(baseRot, targetFootRot, Mathf.Clamp01(weight))
            : targetFootRot;
    }

    public void SetFlexDegrees(float deg) => flexInput = inputIsRadians ? deg * Mathf.Deg2Rad : deg;
    public void SetCalfRotationDegrees(float deg) => calfRotInput = inputIsRadians ? deg * Mathf.Deg2Rad : deg;

    static Vector3 GetAxisWorld(Transform space, Vector3 localAxis, Vector3 fallbackLocalAxis)
    {
        Vector3 axisLocal = localAxis.sqrMagnitude > 0.0001f ? localAxis : fallbackLocalAxis;
        axisLocal.Normalize();
        return space.TransformDirection(axisLocal);
    }
}
