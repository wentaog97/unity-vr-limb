using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// UI Controller that bridges MyoConnectionManager with UI dropdowns and displays.
/// Handles device selection via dropdown and updates UI based on connection state.
/// 
/// Setup:
/// 1. Attach to a GameObject in your scene
/// 2. Assign MyoConnectionManager reference
/// 3. Assign UI elements (dropdown, buttons, text displays)
/// 4. Buttons will automatically connect to the device selected in the dropdown
/// </summary>
public class MyoUIController : MonoBehaviour
{
    [Header("Manager Reference")]
    [Tooltip("Reference to the MyoConnectionManager")]
    public MyoConnectionManager myoManager;
    
    [Header("Device Selection")]
    [Tooltip("Dropdown for selecting discovered devices")]
    public TMP_Dropdown deviceDropdown;
    
    [Header("Control Buttons")]
    public Button scanButton;
    public Button connectButton;
    public Button disconnectButton;
    public Button vibrateButton;
    
    [Header("Status Displays")]
    public TextMeshProUGUI selectedDeviceText;
    public TextMeshProUGUI connectionStatusText;
    public TextMeshProUGUI batteryLevelText;
    public TextMeshProUGUI connectedDevicesText;
    public TextMeshProUGUI outputLogText;
    
    [Header("Settings")]
    [Tooltip("Auto-select first device after scan")]
    public bool autoSelectFirstDevice = true;
    
    // Private state
    private string currentlySelectedAddress = "";
    private List<MyoDeviceInfo> cachedDiscoveredDevices = new List<MyoDeviceInfo>();
    
    private void Awake()
    {
        // Find manager if not assigned
        if (!myoManager)
        {
            myoManager = FindFirstObjectByType<MyoConnectionManager>();
            if (!myoManager)
            {
                Debug.LogError("[MyoUI] MyoConnectionManager not found! Please assign it.");
                return;
            }
        }
        
        // Set up button listeners
        if (scanButton) scanButton.onClick.AddListener(OnScanClicked);
        if (connectButton) connectButton.onClick.AddListener(OnConnectClicked);
        if (disconnectButton) disconnectButton.onClick.AddListener(OnDisconnectClicked);
        if (vibrateButton) vibrateButton.onClick.AddListener(OnVibrateClicked);
        
        // Set up dropdown listener
        if (deviceDropdown) deviceDropdown.onValueChanged.AddListener(OnDropdownValueChanged);
        
        // Subscribe to manager events
        if (myoManager)
        {
            myoManager.OnDeviceListUpdated += OnDeviceListUpdated;
            myoManager.OnConnectedDevicesUpdated += OnConnectedDevicesUpdated;
        }
        
        // Initial UI update
        UpdateUI();
    }
    
    private void Update()
    {
        // Continuously update UI to reflect connection state changes
        // This ensures buttons enable/disable properly when server connects/disconnects
        UpdateUI();
    }
    
    private void OnDestroy()
    {
        // Unsubscribe from events
        if (myoManager)
        {
            myoManager.OnDeviceListUpdated -= OnDeviceListUpdated;
            myoManager.OnConnectedDevicesUpdated -= OnConnectedDevicesUpdated;
        }
    }
    
    // ==================== BUTTON HANDLERS ====================
    
    private void OnScanClicked()
    {
        if (myoManager)
        {
            myoManager.ScanForDevices();
            Log("Scanning for Myo armbands...");
        }
    }
    
    private void OnConnectClicked()
    {
        if (myoManager && !string.IsNullOrEmpty(currentlySelectedAddress))
        {
            myoManager.ConnectToDevice(currentlySelectedAddress);
            Log($"Connecting to {currentlySelectedAddress}...");
        }
        else if (string.IsNullOrEmpty(currentlySelectedAddress))
        {
            Log("[ERROR] No device selected. Please select a device from the dropdown.");
        }
    }
    
    private void OnDisconnectClicked()
    {
        if (myoManager && !string.IsNullOrEmpty(currentlySelectedAddress))
        {
            // Check if the selected address is actually connected
            if (myoManager.ConnectedDevices.ContainsKey(currentlySelectedAddress))
            {
                myoManager.DisconnectDevice(currentlySelectedAddress);
                Log($"Disconnecting from {currentlySelectedAddress}...");
            }
            else
            {
                Log("[ERROR] Selected device is not connected.");
            }
        }
        else
        {
            Log("[ERROR] No device selected.");
        }
    }
    
    private void OnVibrateClicked()
    {
        if (myoManager && !string.IsNullOrEmpty(currentlySelectedAddress))
        {
            // Check if the selected address is actually connected
            if (myoManager.ConnectedDevices.ContainsKey(currentlySelectedAddress))
            {
                myoManager.VibrateDevice(currentlySelectedAddress);
                Log($"Sending vibration to {currentlySelectedAddress}...");
            }
            else
            {
                Log("[ERROR] Selected device is not connected. Can only vibrate connected devices.");
            }
        }
        else
        {
            Log("[ERROR] No device selected.");
        }
    }
    
    // ==================== DROPDOWN HANDLER ====================
    
    private void OnDropdownValueChanged(int index)
    {
        if (index >= 0 && index < cachedDiscoveredDevices.Count)
        {
            currentlySelectedAddress = cachedDiscoveredDevices[index].address;
            Log($"Selected: {cachedDiscoveredDevices[index].name} ({currentlySelectedAddress})");
            UpdateUI();
        }
    }
    
    // ==================== EVENT HANDLERS ====================
    
    private void OnDeviceListUpdated()
    {
        // Update the dropdown with discovered devices
        PopulateDropdown();
        UpdateUI();
        
        Log($"Device list updated: {myoManager.DiscoveredDevices.Count} device(s) found");
    }
    
    private void OnConnectedDevicesUpdated()
    {
        UpdateUI();
        Log($"Connected devices updated: {myoManager.ConnectedDeviceCount} device(s) connected");
    }
    
    // ==================== UI UPDATE METHODS ====================
    
    private void PopulateDropdown()
    {
        if (!deviceDropdown) return;
        
        // Cache the discovered devices
        cachedDiscoveredDevices = new List<MyoDeviceInfo>(myoManager.DiscoveredDevices);
        
        // Clear existing options
        deviceDropdown.ClearOptions();
        
        if (cachedDiscoveredDevices.Count == 0)
        {
            // Add placeholder
            deviceDropdown.AddOptions(new List<string> { "No devices found" });
            deviceDropdown.interactable = false;
            currentlySelectedAddress = "";
            deviceDropdown.RefreshShownValue();
            return;
        }
        
        // Create dropdown options
        List<string> options = new List<string>();
        foreach (var device in cachedDiscoveredDevices)
        {
            // Format: "Device Name (XX:XX:XX:XX) RSSI: -XX"
            string optionText = $"{device.name} ({device.address}) RSSI: {device.rssi}";
            options.Add(optionText);
        }
        
        deviceDropdown.AddOptions(options);
        deviceDropdown.interactable = true;
        
        // Auto-select first device if enabled
        if (autoSelectFirstDevice && cachedDiscoveredDevices.Count > 0)
        {
            deviceDropdown.value = 0;
            currentlySelectedAddress = cachedDiscoveredDevices[0].address;
            Log($"Auto-selected: {cachedDiscoveredDevices[0].name}");
        }

        deviceDropdown.RefreshShownValue();
    }
    
    private void UpdateUI()
    {
        if (!myoManager) return;
        
        bool hasSelectedAddress = !string.IsNullOrEmpty(currentlySelectedAddress);
        bool isSelectedConnected = hasSelectedAddress && myoManager.ConnectedDevices.ContainsKey(currentlySelectedAddress);
        bool hasDiscoveredDevices = myoManager.DiscoveredDevices.Count > 0;
        bool hasConnectedDevices = myoManager.ConnectedDeviceCount > 0;
        
        // Update button states
        if (scanButton)
        {
            scanButton.interactable = myoManager.netClient && myoManager.netClient.IsConnected;
        }
        
        if (connectButton)
        {
            // Can connect if: have selected device AND it's not already connected
            connectButton.interactable = hasSelectedAddress && !isSelectedConnected;
        }
        
        if (disconnectButton)
        {
            // Can disconnect if: have selected device AND it IS connected
            disconnectButton.interactable = isSelectedConnected;
        }
        
        if (vibrateButton)
        {
            // Can vibrate if: have selected device AND it IS connected
            vibrateButton.interactable = isSelectedConnected;
        }
        
        // Update selected device text
        if (selectedDeviceText)
        {
            if (!hasSelectedAddress)
            {
                selectedDeviceText.text = "No device selected";
            }
            else
            {
                var discoveredDevice = cachedDiscoveredDevices.Find(d => d.address == currentlySelectedAddress);
                if (discoveredDevice != null)
                {
                    selectedDeviceText.text = $"Selected: {discoveredDevice.name}\n{currentlySelectedAddress}";
                }
                else
                {
                    selectedDeviceText.text = $"Selected: {currentlySelectedAddress}";
                }
            }
        }
        
        // Update connection status
        if (connectionStatusText)
        {
            if (!hasSelectedAddress)
            {
                connectionStatusText.text = "Status: No device selected";
            }
            else if (isSelectedConnected)
            {
                var device = myoManager.ConnectedDevices[currentlySelectedAddress];
                connectionStatusText.text = $"Status: Connected\nDevice: {device.name}";
            }
            else
            {
                connectionStatusText.text = "Status: Not connected";
            }
        }
        
        // Update battery level
        if (batteryLevelText)
        {
            if (isSelectedConnected)
            {
                var device = myoManager.ConnectedDevices[currentlySelectedAddress];
                if (device.battery >= 0)
                {
                    batteryLevelText.text = $"Battery: {device.battery}%";
                }
                else
                {
                    batteryLevelText.text = "Battery: Unknown";
                }
            }
            else
            {
                batteryLevelText.text = "Battery: --";
            }
        }
        
        // Update connected devices list
        if (connectedDevicesText)
        {
            if (hasConnectedDevices)
            {
                string deviceList = $"Connected Devices ({myoManager.ConnectedDeviceCount}):\n";
                foreach (var device in myoManager.ConnectedDevices.Values)
                {
                    string batteryStr = device.battery >= 0 ? $"{device.battery}%" : "N/A";
                    deviceList += $"• {device.name} - Battery: {batteryStr}\n";
                }
                connectedDevicesText.text = deviceList.TrimEnd('\n');
            }
            else
            {
                connectedDevicesText.text = "No devices connected";
            }
        }
    }
    
    private void Log(string message)
    {
        Debug.Log($"[MyoUI] {message}");
        
        if (outputLogText)
        {
            // Add timestamp
            string timestamp = System.DateTime.Now.ToString("HH:mm:ss");
            outputLogText.text += $"[{timestamp}] {message}\n";
            
            // Keep last 20 lines to prevent overflow
            var lines = outputLogText.text.Split('\n');
            if (lines.Length > 20)
            {
                outputLogText.text = string.Join("\n", lines, lines.Length - 20, 20);
            }
        }
    }
    
    // ==================== PUBLIC HELPER METHODS ====================
    
    /// <summary>
    /// Get the currently selected device address
    /// </summary>
    public string GetSelectedDeviceAddress()
    {
        return currentlySelectedAddress;
    }
    
    /// <summary>
    /// Set the selected device by address (also updates dropdown)
    /// </summary>
    public void SetSelectedDevice(string address)
    {
        int index = cachedDiscoveredDevices.FindIndex(d => d.address == address);
        if (index >= 0 && deviceDropdown)
        {
            deviceDropdown.value = index;
            currentlySelectedAddress = address;
            deviceDropdown.RefreshShownValue();
            UpdateUI();
        }
    }
    
    /// <summary>
    /// Clear the output log
    /// </summary>
    public void ClearLog()
    {
        if (outputLogText)
        {
            outputLogText.text = "";
        }
    }
}
