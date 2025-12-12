using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;

/// <summary>
/// UI Controller for managing server-side EMG inference.
/// Provides buttons and dropdowns for model selection and inference control.
/// </summary>
public class ServerInferenceController : MonoBehaviour
{
    [Header("References")]
    public ServerEMGInterpreter serverInterpreter;
    public NetworkBridgeClient networkClient;
    
    [Header("UI Elements")]
    public Button refreshModelsButton;
    public Button startInferenceButton;
    public Button stopInferenceButton;
    public TMP_Dropdown modelDropdown;
    public TextMeshProUGUI statusText;
    public TextMeshProUGUI connectionText;
    
    [Header("Settings")]
    public bool autoRefreshModelsOnConnect = true;
    
    private bool isInferenceRunning = false;
    private List<string> availableModels = new();
    private bool wasConnected = false;
    
    private void Awake()
    {
        // Find components if not assigned
        if (!serverInterpreter) serverInterpreter = FindFirstObjectByType<ServerEMGInterpreter>();
        if (!networkClient) networkClient = FindFirstObjectByType<NetworkBridgeClient>();
        
        // Setup UI callbacks
        SetupUI();
        
        // Subscribe to network events
        if (networkClient)
        {
            networkClient.ModelListReceived += OnModelListReceived;
            networkClient.ModelSetConfirmed += OnModelSetConfirmed;
            networkClient.ErrorReceived += OnServerError;
        }
        
        UpdateUI();
    }
    
    private void OnDestroy()
    {
        if (networkClient)
        {
            networkClient.ModelListReceived -= OnModelListReceived;
            networkClient.ModelSetConfirmed -= OnModelSetConfirmed;
            networkClient.ErrorReceived -= OnServerError;
        }
    }
    
    private void SetupUI()
    {
        // Refresh models button
        if (refreshModelsButton)
        {
            refreshModelsButton.onClick.AddListener(OnRefreshModelsClicked);
        }
        
        // Start inference button
        if (startInferenceButton)
        {
            startInferenceButton.onClick.AddListener(OnStartInferenceClicked);
        }
        
        // Stop inference button
        if (stopInferenceButton)
        {
            stopInferenceButton.onClick.AddListener(OnStopInferenceClicked);
        }
        
        // Model dropdown
        if (modelDropdown)
        {
            modelDropdown.onValueChanged.AddListener(OnModelSelected);
        }
        
    }
    
    private void Update()
    {
        UpdateUI();
    }
    
    // ==================== UI EVENT HANDLERS ====================
    
    private void OnRefreshModelsClicked()
    {
        if (serverInterpreter)
        {
            serverInterpreter.RequestModelList();
            UpdateStatusText("Requesting model list...");
        }
    }
    
    private void OnStartInferenceClicked()
    {
        if (serverInterpreter && !isInferenceRunning)
        {
            // Set model if one is selected
            if (modelDropdown && modelDropdown.value < availableModels.Count)
            {
                string selectedModel = availableModels[modelDropdown.value];
                serverInterpreter.SetModel(selectedModel);
            }
            
            serverInterpreter.StartInterpretation();
            isInferenceRunning = true;
            UpdateStatusText("Starting inference...");
        }
    }
    
    private void OnStopInferenceClicked()
    {
        if (serverInterpreter && isInferenceRunning)
        {
            serverInterpreter.StopInterpretation();
            isInferenceRunning = false;
            UpdateStatusText("Stopping inference...");
        }
    }
    
    private void OnModelSelected(int index)
    {
        if (index < availableModels.Count && serverInterpreter)
        {
            string selectedModel = availableModels[index];
            serverInterpreter.SetModel(selectedModel);
            UpdateStatusText($"Setting model to {selectedModel}...");
        }
    }
    
    // ==================== NETWORK EVENT HANDLERS ====================
    
    private void OnModelListReceived(string modelListStr)
    {
        availableModels.Clear();
        
        if (modelListStr != "NONE" && !string.IsNullOrEmpty(modelListStr))
        {
            string[] models = modelListStr.Split(',');
            availableModels.AddRange(models);
        }
        
        UpdateModelDropdown();
        UpdateStatusText($"Found {availableModels.Count} models");
    }
    
    private void OnModelSetConfirmed(string modelName)
    {
        UpdateStatusText($"Model set to: {modelName}");
    }
    
    private void OnServerError(string error)
    {
        UpdateStatusText($"Error: {error}");
    }
    
    // ==================== UI UPDATE METHODS ====================
    
    private void UpdateUI()
    {
        bool connected = networkClient && networkClient.IsConnected;
        if (connected && !wasConnected && autoRefreshModelsOnConnect)
        {
            OnRefreshModelsClicked();
        }
        wasConnected = connected;
        bool hasModels = availableModels.Count > 0;
        
        // Update connection status
        if (connectionText)
        {
            connectionText.text = connected ? "Connected" : "Disconnected";
            connectionText.color = connected ? Color.green : Color.red;
        }
        
        // Update button states
        if (refreshModelsButton)
            refreshModelsButton.interactable = connected;
        
        if (startInferenceButton)
            startInferenceButton.interactable = connected && hasModels && !isInferenceRunning;
        
        if (stopInferenceButton)
            stopInferenceButton.interactable = connected && isInferenceRunning;
        
        if (modelDropdown)
            modelDropdown.interactable = connected && hasModels && !isInferenceRunning;
        
        // Update inference running status
        if (serverInterpreter)
        {
            isInferenceRunning = serverInterpreter.IsRunning;
        }
    }
    
    private void UpdateModelDropdown()
    {
        if (modelDropdown)
        {
            modelDropdown.ClearOptions();
            
            if (availableModels.Count > 0)
            {
                modelDropdown.AddOptions(availableModels);
            }
            else
            {
                modelDropdown.AddOptions(new List<string> { "No models available" });
            }
            
            modelDropdown.value = 0;
        }
    }
    
    private void UpdateStatusText(string message)
    {
        if (statusText)
        {
            statusText.text = message;
        }
        Debug.Log($"[ServerInferenceController] {message}");
    }
    
    // ==================== PUBLIC API ====================
    
    /// <summary>
    /// Programmatically select a model by name
    /// </summary>
    public void SelectModel(string modelName)
    {
        int index = availableModels.IndexOf(modelName);
        if (index >= 0 && modelDropdown)
        {
            modelDropdown.value = index;
            OnModelSelected(index);
        }
    }
    
    /// <summary>
    /// Get list of available model names
    /// </summary>
    public List<string> GetAvailableModels()
    {
        return new List<string>(availableModels);
    }
}

