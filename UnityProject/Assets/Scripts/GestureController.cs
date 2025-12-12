using UnityEngine;

/// <summary>
/// Bridges the decoded gesture value from the server to an Animator parameter.
/// Supports the new 1D blend tree where -1 = clench, 0 = rest, 1 = open.
/// </summary>
public class GestureController : MonoBehaviour
{
    [Header("Animator")]
    public Animator animator;

    [Header("Control Mode")]
    [Tooltip("If enabled, EMG interpreters can drive the gesture blend value.")]
    public bool allowEMGControl = true;

    [Header("Blend Settings")]
    [Tooltip("Animator float parameter that the 1D blend tree listens to.")]
    public string blendParameter = "GestureBlend";
    [Range(-1f, 1f)]
    public float gestureBlend = 0f;
    [Tooltip("Optionally mirror the same value into legacy X/Y parameters for backwards compatibility.")]
    public bool mirrorToLegacyXY = false;
    public string legacyXParameter = "x";
    public string legacyYParameter = "y";

    private int _blendHash;
    private int _legacyXHash;
    private int _legacyYHash;

    private void Awake()
    {
        CacheParameterHashes();
    }

    private void OnValidate()
    {
        CacheParameterHashes();
        gestureBlend = Mathf.Clamp(gestureBlend, -1f, 1f);
    }

    private void CacheParameterHashes()
    {
        _blendHash = Animator.StringToHash(blendParameter);
        _legacyXHash = Animator.StringToHash(legacyXParameter);
        _legacyYHash = Animator.StringToHash(legacyYParameter);
    }

    private void Update()
    {
        if (!animator) return;

        animator.SetFloat(_blendHash, gestureBlend);

        if (mirrorToLegacyXY)
        {
            animator.SetFloat(_legacyXHash, gestureBlend);
            animator.SetFloat(_legacyYHash, 0f);
        }


        if (!animator) 
        {
            Debug.LogWarning("Animator is null!");
            return;
        }

        animator.SetFloat(_blendHash, gestureBlend);
        Debug.Log($"Setting {blendParameter} to {gestureBlend}");

        if (mirrorToLegacyXY)
        {
            animator.SetFloat(_legacyXHash, gestureBlend);
            animator.SetFloat(_legacyYHash, 0f);
            Debug.Log($"Setting {legacyXParameter} to {gestureBlend}");
        }
    }

    /// <summary>
    /// Updates the gesture blend parameter if EMG control is enabled.
    /// </summary>
    public void SetGestureBlend(float value)
    {
        if (!allowEMGControl) return;
        gestureBlend = Mathf.Clamp(value, -1f, 1f);
    }
}
