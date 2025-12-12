using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// Handles the guided recording/training workflow exposed by the Python server.
/// Listens for PIPELINE events, updates UI prompts, and forwards GUIDED_* commands via NetworkBridgeClient.
/// </summary>
public class GuidedPipelineUIController : MonoBehaviour
{
    [Header("Dependencies")] public NetworkBridgeClient networkClient;
    public ServerEMGInterpreter serverInterpreter;
    public ServerInferenceController inferenceController;

    [Header("Controls")] public Button startButton;

    [Header("Status UI")] public TextMeshProUGUI sessionText;
    public TextMeshProUGUI instructionsText;
    public Slider segmentProgress;
    public TextMeshProUGUI logText;

    [Header("Automation")]
    [Tooltip("Automatically request LIST_MODELS when training completes.")]
    public bool autoRequestModelsOnComplete = true;
    [Tooltip("Automatically SET_MODEL to the freshly trained session once complete.")]
    public bool autoSelectTrainedModel = true;

    [Header("Auto Status Polling")]
    [Tooltip("Delay before automatically requesting guided status after starting a session (seconds).")]
    public float statusPingDelay = 1.5f;

    private readonly Queue<string> logLines = new();
    private const int MaxLogLines = 60;

    private string currentSessionId = string.Empty;
    private string currentStep = string.Empty;
    private float currentSegmentDuration = 0f;
    private int totalSteps = 0;
    private int stepIndex = 0;
    private Coroutine statusPingRoutine;

    private string GenerateSessionLabel()
    {
        return $"myo_raw_session_{DateTime.Now:yyyyMMdd_HHmmss}";
    }

    private string GetGestureDisplayName()
    {
        return string.IsNullOrEmpty(currentStep) ? "the next gesture" : currentStep;
    }

    private string GetDurationDisplayText()
    {
        return currentSegmentDuration > 0f
            ? $"{currentSegmentDuration:F0} seconds"
            : "a few seconds";
    }

    private void Awake()
    {
        if (!networkClient) networkClient = FindFirstObjectByType<NetworkBridgeClient>();
        if (!serverInterpreter) serverInterpreter = FindFirstObjectByType<ServerEMGInterpreter>();
        if (!inferenceController) inferenceController = FindFirstObjectByType<ServerInferenceController>();

        if (startButton) startButton.onClick.AddListener(OnStartClicked);

        UpdateButtonStates();
    }

    private void OnEnable()
    {
        Subscribe();
    }

    private void OnDisable()
    {
        Unsubscribe();
        StopStatusPing();
    }

    private void OnDestroy()
    {
        Unsubscribe();
        StopStatusPing();
    }

    private void Update()
    {
        UpdateButtonStates();
    }

    private void Subscribe()
    {
        if (networkClient == null) return;
        networkClient.PipelineEventReceived += OnPipelineEvent;
        networkClient.PipelineStatusReceived += OnPipelineStatus;
        networkClient.ErrorReceived += OnServerError;
    }

    private void Unsubscribe()
    {
        if (networkClient == null) return;
        networkClient.PipelineEventReceived -= OnPipelineEvent;
        networkClient.PipelineStatusReceived -= OnPipelineStatus;
        networkClient.ErrorReceived -= OnServerError;
    }

    private void UpdateButtonStates()
    {
        bool connected = networkClient && networkClient.IsConnected;
        if (startButton) startButton.interactable = connected;
    }

    private void OnStartClicked()
    {
        if (networkClient == null)
        {
            AppendLog("[WARN] No NetworkBridgeClient assigned");
            return;
        }

        string label = GenerateSessionLabel();
        networkClient.StartGuidedSession(label);
        if (sessionText) sessionText.text = $"Session: {label}";
        AppendLog($"GUIDED_START sent with label '{label}'");
        UpdateInstructionText("Starting guided session… waiting for first instruction from server.");
        ScheduleStatusPing();
    }

    private void OnPipelineEvent(string json)
    {
        PipelineEventPayload payload = null;
        try
        {
            payload = JsonUtility.FromJson<PipelineEventPayload>(json);
        }
        catch (Exception ex)
        {
            AppendLog($"[ERROR] Failed to parse pipeline event: {ex.Message}");
            return;
        }

        if (payload == null)
        {
            AppendLog("[WARN] Received empty pipeline payload");
            return;
        }

        if (!string.IsNullOrEmpty(payload.session))
        {
            currentSessionId = payload.session;
            if (sessionText) sessionText.text = $"Session: {currentSessionId}";
        }

        switch (payload.type)
        {
            case "session_started":
                totalSteps = payload.total;
                stepIndex = 0;
                currentSegmentDuration = 0f;
                currentStep = string.Empty;
                UpdateInstructionText(totalSteps > 0
                    ? $"Guided session started. We'll capture {totalSteps} gestures."
                    : "Guided session started. Follow the on-screen instructions.");
                AppendLog($"Session {currentSessionId} started ({totalSteps} steps)");
                break;

            case "countdown_ready":
                totalSteps = payload.steps;
                stepIndex = 0;
                UpdateInstructionText(totalSteps > 0
                    ? $"Get ready. We'll guide you through {totalSteps} gestures."
                    : "Get ready for the next recording sequence.");
                break;

            case "step_prompt":
                stepIndex = payload.index;
                currentStep = payload.step;
                currentSegmentDuration = payload.duration;
                UpdateInstructionText(
                    $"We will begin with {GetGestureDisplayName()} for {GetDurationDisplayText()}. Start and hold the gesture when prompted.");
                AppendLog($"Prompt: {payload.prompt}");
                break;

            case "countdown":
                {
                    float remaining = Mathf.Max(0f, payload.remaining);
                    string countdownMessage = remaining > 0f
                        ? $"Starting in {Mathf.CeilToInt(remaining)} seconds."
                        : "Starting now.";
                    UpdateInstructionText(
                        $"We will begin with {GetGestureDisplayName()} for {GetDurationDisplayText()}. {countdownMessage}");
                }
                break;

            case "segment_started":
                currentStep = payload.step;
                currentSegmentDuration = payload.duration;
                if (segmentProgress) segmentProgress.value = 0f;
                UpdateInstructionText(
                    $"Start and hold your {GetGestureDisplayName()} gesture for {GetDurationDisplayText()}.");
                break;

            case "segment_tick":
                if (segmentProgress && currentSegmentDuration > 0f)
                {
                    float progress = Mathf.Clamp01(payload.elapsed / currentSegmentDuration);
                    segmentProgress.value = progress;
                }
                if (currentSegmentDuration > 0f)
                {
                    float remainingTime = Mathf.Max(0f, currentSegmentDuration - payload.elapsed);
                    string remainingMessage = remainingTime > 0f
                        ? $"{remainingTime:F1}s remaining."
                        : "Almost done.";
                    UpdateInstructionText($"Hold your {GetGestureDisplayName()} gesture. {remainingMessage}");
                }
                else
                {
                    UpdateInstructionText($"Hold your {GetGestureDisplayName()} gesture.");
                }
                break;

            case "segment_completed":
                if (segmentProgress) segmentProgress.value = 1f;
                UpdateInstructionText($"Great! {GetGestureDisplayName()} captured. Relax until the next cue.");
                break;

            case "rest":
                UpdateInstructionText("Rest while we prepare the next gesture.");
                if (segmentProgress) segmentProgress.value = 0f;
                break;

            case "recording_complete":
                UpdateInstructionText("Recording complete. Processing data...");
                AppendLog("Recording complete – starting feature extraction");
                break;

            case "training_started":
                UpdateInstructionText(payload.samples > 0
                    ? $"Training models with {payload.samples} samples..."
                    : "Training models...");
                AppendLog($"Training models ({payload.samples} samples)");
                break;

            case "training_complete":
                UpdateInstructionText("Training complete. Reviewing results...");
                AppendLog(payload.model != null
                    ? $"Training complete – model '{payload.model}' ready"
                    : "Training complete");

                if (autoRequestModelsOnComplete)
                {
                    serverInterpreter?.RequestModelList();
                }

                if (autoSelectTrainedModel && !string.IsNullOrEmpty(payload.model))
                {
                    serverInterpreter?.SetModel(payload.model);
                    inferenceController?.SelectModel(payload.model);
                }

                if (payload.metrics != null)
                {
                    AppendLog($"Metrics: windows={payload.metrics.window_count}, gesture_windows={payload.metrics.gesture_window_count}");
                }
                break;

            case "session_cancelled":
                UpdateInstructionText("Session cancelled. You can start again whenever you're ready.");
                AppendLog("Session cancelled by user");
                break;

            case "session_error":
                UpdateInstructionText("Session error. Check the log for details.");
                AppendLog($"[ERROR] {payload.error}");
                break;

            default:
                AppendLog($"Event: {payload.type} {json}");
                break;
        }
    }

    private void OnPipelineStatus(string json)
    {
        PipelineStatusPayload status = null;
        try
        {
            status = JsonUtility.FromJson<PipelineStatusPayload>(json);
        }
        catch (Exception ex)
        {
            AppendLog($"[ERROR] Failed to parse pipeline status: {ex.Message}");
            return;
        }

        if (status == null) return;

        if (!string.IsNullOrEmpty(status.session))
        {
            currentSessionId = status.session;
            if (sessionText) sessionText.text = $"Session: {currentSessionId}";
        }

        if (!string.IsNullOrEmpty(status.state))
        {
            UpdateInstructionText($"Status: {status.state}");
        }

        if (!string.IsNullOrEmpty(status.step))
        {
            currentStep = status.step;
        }
    }

    private void OnServerError(string error)
    {
        AppendLog($"[ERROR] {error}");
        UpdateInstructionText("Error received from server. Check the log for details.");
    }

    private void UpdateInstructionText(string message)
    {
        if (instructionsText) instructionsText.text = message;
        if (statusPingRoutine != null)
        {
            StopCoroutine(statusPingRoutine);
            statusPingRoutine = null;
        }
    }

    private void ScheduleStatusPing()
    {
        if (!networkClient || statusPingDelay <= 0f) return;
        StopStatusPing();
        statusPingRoutine = StartCoroutine(RequestStatusAfterDelay(statusPingDelay));
    }

    private void StopStatusPing()
    {
        if (statusPingRoutine != null)
        {
            StopCoroutine(statusPingRoutine);
            statusPingRoutine = null;
        }
    }

    private IEnumerator RequestStatusAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        networkClient?.RequestGuidedStatus();
        statusPingRoutine = null;
    }

    private void AppendLog(string message)
    {
        string timestamped = $"[{DateTime.Now:HH:mm:ss}] {message}";
        logLines.Enqueue(timestamped);
        while (logLines.Count > MaxLogLines)
        {
            logLines.Dequeue();
        }

        if (logText)
        {
            logText.text = string.Join("\n", logLines);
        }

        Debug.Log($"[PipelineUI] {message}");
    }

    [Serializable]
    private class PipelineEventPayload
    {
        public string type;
        public string session;
        public string step;
        public string prompt;
        public float duration;
        public float elapsed;
        public int index;
        public int total;
        public int steps;
        public float remaining;
        public string model;
        public string error;
        public PipelineMetrics metrics;
        public int samples;
    }

    [Serializable]
    private class PipelineMetrics
    {
        public int window_count;
        public int gesture_window_count;
    }

    [Serializable]
    private class PipelineStatusPayload
    {
        public string session;
        public string state;
        public string step;
        public string model;
    }
}
