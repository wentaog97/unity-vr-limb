using System.Collections.Generic;
using System.Linq;
using UnityEngine;

/// <summary>
/// Multi-armband Myo connection manager (Pure Logic Layer - No UI).
/// Responsibilities:
/// - Manage device discovery and connection state
/// - Track discovered and connected devices
/// - Provide public API for device operations
/// - Fire events for state changes (UI subscribes to these)
/// 
/// This is a pure data/logic layer with no UI dependencies.
/// Use MyoUIController for UI handling.
/// </summary>
public class MyoConnectionManager : MonoBehaviour
{
    [Header("References")]
    public NetworkBridgeClient netClient;
    
    // ==================== PUBLIC DATA FOR UI BINDING ====================
    
    /// <summary>
    /// Devices discovered from last scan (for dropdown/list)
    /// </summary>
    public List<MyoDeviceInfo> DiscoveredDevices { get; private set; } = new List<MyoDeviceInfo>();
    
    /// <summary>
    /// Currently connected devices (key = address)
    /// </summary>
    public Dictionary<string, ConnectedDeviceInfo> ConnectedDevices { get; private set; } = new Dictionary<string, ConnectedDeviceInfo>();
    
    // ==================== EVENTS FOR UI UPDATES ====================
    
    /// <summary>
    /// Fired when discovered devices list changes (after scan)
    /// </summary>
    public event System.Action OnDeviceListUpdated;
    
    /// <summary>
    /// Fired when a device connects or disconnects
    /// </summary>
    public event System.Action OnConnectedDevicesUpdated;
    
    // ==================== PRIVATE STATE ====================
    
    private bool isScanning = false;
    private readonly object stateLock = new object();
    private bool deviceListDirty = false;
    private bool connectedDevicesDirty = false;
    
    // ==================== UNITY LIFECYCLE ====================
    
    private void Awake()
    {
        // Find network client if not assigned
        if (!netClient) netClient = FindFirstObjectByType<NetworkBridgeClient>();
        
        // Subscribe to network events
        if (netClient)
        {
            netClient.MyoDevicesReceived += OnMyoDevicesReceived;
            netClient.MyoConnectedReceived += OnMyoConnected;
            netClient.MyoDisconnectedReceived += OnMyoDisconnected;
            netClient.MyoStatusReceived += OnMyoStatus;
            netClient.MyoErrorReceived += OnMyoError;
        }
    }
    
    private void OnDestroy()
    {
        // Unsubscribe from events
        if (netClient)
        {
            netClient.MyoDevicesReceived -= OnMyoDevicesReceived;
            netClient.MyoConnectedReceived -= OnMyoConnected;
            netClient.MyoDisconnectedReceived -= OnMyoDisconnected;
            netClient.MyoStatusReceived -= OnMyoStatus;
            netClient.MyoErrorReceived -= OnMyoError;
        }
    }
    
    // ==================== PUBLIC API ====================
    
    /// <summary>
    /// Scan for Myo armbands (populates DiscoveredDevices)
    /// </summary>
    public void ScanForDevices()
    {
        if (netClient && netClient.IsConnected && !isScanning)
        {
            netClient.ScanMyo();
            isScanning = true;
            Debug.Log("[MyoConnectionManager] Scanning for Myo devices...");
        }
        else if (!netClient || !netClient.IsConnected)
        {
            Debug.LogWarning("[MyoConnectionManager] Not connected to server");
        }
    }
    
    /// <summary>
    /// Connect to a specific device (moves from discovered to connected)
    /// </summary>
    public void ConnectToDevice(string address)
    {
        if (netClient && netClient.IsConnected && !string.IsNullOrEmpty(address))
        {
            netClient.ConnectMyo(address);
            Debug.Log($"[MyoConnectionManager] Connecting to {address}...");
        }
    }
    
    /// <summary>
    /// Disconnect a specific device
    /// </summary>
    public void DisconnectDevice(string address)
    {
        if (netClient && netClient.IsConnected && !string.IsNullOrEmpty(address))
        {
            netClient.DisconnectMyo(address);
            Debug.Log($"[MyoConnectionManager] Disconnecting from {address}...");
        }
    }
    
    /// <summary>
    /// Send vibration to a device (for identification)
    /// </summary>
    public void VibrateDevice(string address)
    {
        if (netClient && netClient.IsConnected && !string.IsNullOrEmpty(address))
        {
            netClient.VibrateDevice(address);
            Debug.Log($"[MyoConnectionManager] Sending vibration to {address}...");
        }
    }
    
    /// <summary>
    /// Get all connected device addresses
    /// </summary>
    public List<string> GetConnectedAddresses()
    {
        return ConnectedDevices.Keys.ToList();
    }
    
    /// <summary>
    /// Check if any devices are connected
    /// </summary>
    public bool HasConnectedDevices => ConnectedDevices.Count > 0;
    
    /// <summary>
    /// Get count of connected devices
    /// </summary>
    public int ConnectedDeviceCount => ConnectedDevices.Count;
    
    // ==================== NETWORK EVENT HANDLERS ====================
    
    private void OnMyoDevicesReceived(string deviceList)
    {
    isScanning = false;
    var parsedDevices = new List<MyoDeviceInfo>();
        
        if (string.IsNullOrEmpty(deviceList) || deviceList == "NONE")
        {
            Debug.Log("[MyoConnectionManager] No Myo armbands found");
            lock (stateLock)
            {
                DiscoveredDevices.Clear();
                deviceListDirty = true;
            }
            return;
        }
        
        // Parse: "addr1:name1:rssi1|addr2:name2:rssi2|..."
        string[] devices = deviceList.Split('|');
        foreach (string device in devices)
        {
            if (string.IsNullOrWhiteSpace(device)) continue;
            string[] parts = device.Split(':');
            if (parts.Length >= 2)
            {
                var deviceInfo = new MyoDeviceInfo
                {
                    address = parts[0].Trim(),
                    name = parts[1].Trim(),
                    rssi = parts.Length > 2 ? parts[2].Trim() : "N/A"
                };
                parsedDevices.Add(deviceInfo);
                Debug.Log($"[MyoConnectionManager] Found: {deviceInfo.name} ({deviceInfo.address}) RSSI: {deviceInfo.rssi}");
            }
        }
        
        Debug.Log($"[MyoConnectionManager] Scan complete: {parsedDevices.Count} device(s) found");
        lock (stateLock)
        {
            DiscoveredDevices.Clear();
            DiscoveredDevices.AddRange(parsedDevices);
            deviceListDirty = true;
        }
    }
    
    private void OnMyoConnected(string connInfo)
    {
        // Parse: "addr:name:battery"
        string[] parts = connInfo.Split(':');
        if (parts.Length >= 2)
        {
            string address = parts[0].Trim();
            string name = parts[1].Trim();
            int battery = parts.Length > 2 && int.TryParse(parts[2].Trim(), out int b) ? b : -1;
            
            // Add to connected devices
            var deviceInfo = new ConnectedDeviceInfo
            {
                address = address,
                name = name,
                battery = battery,
                connected = true,
                streaming = false
            };
            
            ConnectedDevices[address] = deviceInfo;
            
            Debug.Log($"[MyoConnectionManager] Connected: {name} ({address}) Battery: {battery}%");
            lock (stateLock)
            {
                connectedDevicesDirty = true;
            }
        }
    }
    
    private void OnMyoDisconnected(string address)
    {
        if (ConnectedDevices.ContainsKey(address))
        {
            string name = ConnectedDevices[address].name;
            ConnectedDevices.Remove(address);
            
            Debug.Log($"[MyoConnectionManager] Disconnected: {name} ({address})");
            lock (stateLock)
            {
                connectedDevicesDirty = true;
            }
        }
    }
    
    private void OnMyoStatus(string statusJson)
    {
        try
        {
            // Parse JSON status for a device
            // Format: {"address":"...","name":"...","connected":true,"battery":85,"streaming":false}
            
            // Simple parsing (Unity's JsonUtility doesn't handle root objects well without a wrapper)
            if (statusJson.Contains("\"address\":"))
            {
                // Extract address
                int addrStart = statusJson.IndexOf("\"address\":\"") + 11;
                int addrEnd = statusJson.IndexOf("\"", addrStart);
                string address = statusJson.Substring(addrStart, addrEnd - addrStart);
                
                // Extract battery if present
                if (statusJson.Contains("\"battery\":") && ConnectedDevices.ContainsKey(address))
                {
                    int batteryStart = statusJson.IndexOf("\"battery\":") + 10;
                    int batteryEnd = statusJson.IndexOfAny(new char[] { ',', '}' }, batteryStart);
                    string batteryStr = statusJson.Substring(batteryStart, batteryEnd - batteryStart).Trim();
                    if (int.TryParse(batteryStr, out int battery))
                    {
                        ConnectedDevices[address].battery = battery;
                        Debug.Log($"[MyoConnectionManager] Battery updated for {address}: {battery}%");
                        lock (stateLock)
                        {
                            connectedDevicesDirty = true;
                        }
                    }
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[MyoConnectionManager] Failed to parse status: {e.Message}");
        }
    }
    
    private void OnMyoError(string error)
    {
        Debug.LogError($"[MyoConnectionManager] Myo error: {error}");
    }

    private void Update()
    {
        bool shouldNotifyDevices = false;
        bool shouldNotifyConnections = false;

        lock (stateLock)
        {
            if (deviceListDirty)
            {
                deviceListDirty = false;
                shouldNotifyDevices = true;
            }

            if (connectedDevicesDirty)
            {
                connectedDevicesDirty = false;
                shouldNotifyConnections = true;
            }
        }

        if (shouldNotifyDevices)
        {
            OnDeviceListUpdated?.Invoke();
        }

        if (shouldNotifyConnections)
        {
            OnConnectedDevicesUpdated?.Invoke();
        }
    }

}

// ==================== DATA CLASSES ====================

/// <summary>
/// Info about a discovered (not yet connected) Myo device
/// </summary>
[System.Serializable]
public class MyoDeviceInfo
{
    public string address = string.Empty;
    public string name = string.Empty;
    public string rssi = string.Empty;
    
    public override string ToString()
    {
        return $"{name} ({address}) RSSI: {rssi}";
    }
}

/// <summary>
/// Info about a connected Myo device
/// </summary>
[System.Serializable]
public class ConnectedDeviceInfo
{
    public string address = string.Empty;
    public string name = string.Empty;
    public int battery = -1;          // -1 if unknown
    public bool connected;
    public bool streaming;
    
    public override string ToString()
    {
        string batteryStr = battery >= 0 ? $"{battery}%" : "N/A";
        return $"{name} ({address}) Battery: {batteryStr}";
    }
}
