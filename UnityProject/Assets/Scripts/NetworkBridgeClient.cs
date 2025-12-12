using System;
using System.Text;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Net;
using System.Collections.Concurrent;
using System.IO;

/// <summary>
/// Simple TCP client for Quest side to communicate with Python server.
/// Attach this script to an empty GameObject in your VR scene.
/// Hook up UI references via the Inspector (connectButton, pingButton, statusText, outputText).
/// </summary>
public class NetworkBridgeClient : MonoBehaviour
{
    [Header("Server Connection Settings")]
    public string serverIP = "192.168.1.17"; 
    public int serverPort = 5005;

    [Header("UI References")]
    public Button connectButton;
    public Button pingButton;
    public TextMeshProUGUI statusText;
    public TextMeshProUGUI outputText;

    private TcpClient client;
    private NetworkStream netStream;
    private CancellationTokenSource cts;
    private readonly ConcurrentQueue<string> outgoingQueue = new();
    private Task writerTask;

    // Thread-safe queue for incoming server lines
    private readonly ConcurrentQueue<string> messageQueue = new();
    private readonly ConcurrentQueue<Action> mainThreadActions = new();
    private int mainThreadId;

    // Server response events
    public event System.Action<string, float> PredictionReceived;
    public event System.Action<string> ModelListReceived;
    public event System.Action<string> ModelSetConfirmed;
    public event System.Action<string> ErrorReceived;
    public event System.Action<string> PredictionDataReceived; // New: Combined EMG + AprilTag data
    public event System.Action<string> AprilTagStatusReceived;
    public event System.Action<string> PipelineEventReceived;   // PIPELINE {json}
    public event System.Action<string> PipelineStatusReceived;  // PIPELINE_STATUS {json}
    
    // Myo BLE events (Phase 2)
    public event System.Action<string> MyoDevicesReceived;        // MYO_DEVICES addr1:name1:rssi,...
    public event System.Action<string> MyoConnectedReceived;      // MYO_CONNECTED addr:name:battery
    public event System.Action<string> MyoDisconnectedReceived;   // MYO_DISCONNECTED addr
    public event System.Action<string> MyoStatusReceived;         // MYO_STATUS {json}
    public event System.Action<string> MyoDevicesListReceived;    // MYO_DEVICES_LIST {json}
    public event System.Action<string> MyoErrorReceived;          // MYO_ERROR message

    private string leftover = string.Empty; // stores an incomplete line between chunks
    private string persistentDir; // cached path set on main thread

    private void Awake()
    {
        mainThreadId = Thread.CurrentThread.ManagedThreadId;
        Debug.Log($"[NetworkBridge] Server IP in inspector: {serverIP}");
        AppendOutput($"Server IP set to {serverIP}:{serverPort}");
        connectButton.onClick.AddListener(OnConnectClicked);
        pingButton.onClick.AddListener(OnPingClicked);
        pingButton.interactable = false;
        SetStatus("Disconnected");

        // Cache persistentDataPath (Unity API call must be on main thread)
        persistentDir = Application.persistentDataPath;
    }

    private void OnDestroy()
    {
        CloseConnection();
    }

    private void OnConnectClicked()
    {
        if (client == null || !client.Connected)
        {
            _ = ConnectAsync();
        }
    }

    private void OnPingClicked()
    {
        if (client != null && client.Connected)
        {
            QueueMessage("PING");
            AppendOutput("Ping sent...");
        }
    }

    private async Task ConnectAsync()
    {
        try
        {
            AppendOutput($"Attempting to connect to {serverIP}:{serverPort} ...");
            SetStatus("Connecting...");

            // Parse IP to ensure IPv4 format
            if (!IPAddress.TryParse(serverIP, out IPAddress ipAddress))
            {
                AppendOutput($"Invalid IP Address: {serverIP}");
                SetStatus("Invalid IP");
                return;
            }

            client = new TcpClient(AddressFamily.InterNetwork);
            var connectTask = client.ConnectAsync(ipAddress, serverPort);
            var timeoutTask = Task.Delay(5000); // 5-sec timeout

            var completedTask = await Task.WhenAny(connectTask, timeoutTask);
            if (completedTask == timeoutTask)
            {
                AppendOutput("Connection attempt timed out.");
                SetStatus("Timeout");
                client.Close();
                return;
            }

            if (!client.Connected)
            {
                AppendOutput("Failed to connect (unknown reason).");
                SetStatus("Failed");
                return;
            }

            netStream = client.GetStream();
            AppendOutput("Connected to server!");

            SetStatus("Connected");
            pingButton.interactable = true;
            connectButton.interactable = false;

            // Start background listener
            cts = new CancellationTokenSource();
            _ = Task.Run(() => ListenForMessages(cts.Token));

            // Start writer loop
            writerTask = Task.Run(() => WriterLoop(cts.Token));
        }
        catch (SocketException se)
        {
            AppendOutput($"SocketException: {se.SocketErrorCode}");
            AppendOutput(se.Message);
            Debug.LogError(se);
            SetStatus("Socket Error");
        }
        catch (Exception e)
        {
            AppendOutput($"Exception: {e.Message}");
            Debug.LogError(e);
            SetStatus("Connection Failed");
        }
    }

    private async Task ListenForMessages(CancellationToken token)
    {
        byte[] buffer = new byte[1024];
        try
        {
            while (!token.IsCancellationRequested)
            {
                int bytesRead = await netStream.ReadAsync(buffer, 0, buffer.Length, token);
                if (bytesRead == 0)
                    {
                    // Remote closed
                    throw new Exception("Server closed connection");
                }
                string msg = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                Debug.Log("[NB] chunk len " + msg.Length);
                ProcessIncoming(msg);
            }
        }
        catch (Exception e) when (!(e is OperationCanceledException))
        {
            AppendOutput($"Listen error: {e.Message}");
            Debug.LogError(e);
            SetStatus("Listen Error");
        }
    }

    private async void SendMessageToServer(string message)
    {
        if (netStream == null) return;
        byte[] data = Encoding.UTF8.GetBytes(message);
        try
        {
            await netStream.WriteAsync(data, 0, data.Length);
            await netStream.FlushAsync();
            Debug.Log($"[PIPELINE] Sent {data.Length} bytes of JSON");
        }
        catch (Exception e)
        {
            Debug.LogError($"Send error: {e.Message}");
            SetStatus("Disconnected");
        }
    }

    private void CloseConnection()
    {
        try
        {
            cts?.Cancel();
            netStream?.Close();
            client?.Close();
        }
        catch { }
        finally
        {
            SetStatus("Disconnected");
        }
    }

    private void SetStatus(string text)
    {
        RunOnMainThread(() =>
        {
            if (statusText) statusText.text = text;
        });
    }

    private void AppendOutput(string line)
    {
        RunOnMainThread(() =>
        {
            if (outputText)
            {
                outputText.text += line + "\n";
            }
        });
    }

    private void Update()
    {
        while (mainThreadActions.TryDequeue(out var action))
        {
            try
            {
                action?.Invoke();
            }
            catch (Exception ex)
            {
                Debug.LogError($"[NetworkBridge] Main-thread action failed: {ex.Message}");
            }
        }

        while (messageQueue.TryDequeue(out var msg))
        {
            AppendOutput($"Received message: {msg}");
        }
    }

    private void RunOnMainThread(Action action)
    {
        if (action == null)
        {
            return;
        }

        if (Thread.CurrentThread.ManagedThreadId == mainThreadId)
        {
            action();
        }
        else
        {
            mainThreadActions.Enqueue(action);
        }
    }

    private async Task WriterLoop(CancellationToken token)
    {
        try
        {
            Debug.Log("[NetworkBridge] Writer loop started");
            while (!token.IsCancellationRequested)
            {
                if (outgoingQueue.TryDequeue(out var line))
                {
                    if (!line.EndsWith("\n")) line += "\n";
                    byte[] data = Encoding.UTF8.GetBytes(line);
                    await netStream.WriteAsync(data, 0, data.Length, token);
                    await netStream.FlushAsync(token);
                    Debug.Log($"[NetworkBridge] Sent to server: {line.TrimEnd()}");
                }
                else
                {
                    await Task.Delay(5, token);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[NetworkBridge] Writer loop error: {e.Message}");
        }
        finally
        {
            Debug.Log("[NetworkBridge] Writer loop stopped");
        }
    }

    public bool IsConnected => client != null && client.Connected;

    public void QueueMessage(string line)
    {
        if (IsConnected)
        {
            outgoingQueue.Enqueue(line);
            Debug.Log($"[NetworkBridge] Queued message: {line} (Queue size: {outgoingQueue.Count})");
        }
        else
        {
            Debug.LogWarning($"[NetworkBridge] Cannot queue message '{line}' - not connected!");
        }
    }

    #region Guided Session Helpers

    public void StartGuidedSession(string sessionLabel = "")
    {
        if (!IsConnected)
        {
            Debug.LogWarning("[NetworkBridge] Cannot start guided session – not connected to server.");
            return;
        }

        string command = string.IsNullOrWhiteSpace(sessionLabel)
            ? "GUIDED_START"
            : $"GUIDED_START,{sessionLabel.Trim()}";
        QueueMessage(command);
    }

    public void AbortGuidedSession()
    {
        if (!IsConnected)
        {
            Debug.LogWarning("[NetworkBridge] Cannot abort guided session – not connected to server.");
            return;
        }
        QueueMessage("GUIDED_ABORT");
    }

    public void RequestGuidedStatus()
    {
        if (!IsConnected)
        {
            Debug.LogWarning("[NetworkBridge] Cannot request guided status – not connected to server.");
            return;
        }
        QueueMessage("GUIDED_STATUS");
    }

    #endregion
    
    // =============== MYO BLE COMMANDS (Phase 2) ===============
    
    /// <summary>
    /// Scan for Myo armbands via BLE
    /// </summary>
    public void ScanMyo()
    {
        if (IsConnected)
        {
            QueueMessage("SCAN_MYO");
            Debug.Log("[NetworkBridge] Scanning for Myo armbands...");
        }
    }
    
    /// <summary>
    /// Connect to a Myo armband (auto-connects to first found if no address provided)
    /// </summary>
    public void ConnectMyo(string address = "")
    {
        if (IsConnected)
        {
            if (string.IsNullOrEmpty(address))
            {
                QueueMessage("CONNECT_MYO");
                Debug.Log("[NetworkBridge] Connecting to Myo (auto)...");
            }
            else
            {
                QueueMessage($"CONNECT_MYO,{address}");
                Debug.Log($"[NetworkBridge] Connecting to Myo: {address}");
            }
        }
    }
    
    /// <summary>
    /// Disconnect from a Myo armband
    /// </summary>
    public void DisconnectMyo(string address)
    {
        if (IsConnected)
        {
            QueueMessage($"DISCONNECT_MYO,{address}");
            Debug.Log($"[NetworkBridge] Disconnecting from Myo: {address}");
        }
    }
    
    /// <summary>
    /// List all connected Myo armbands
    /// </summary>
    public void ListMyoDevices()
    {
        if (IsConnected)
        {
            QueueMessage("LIST_MYO_DEVICES");
            Debug.Log("[NetworkBridge] Requesting Myo device list...");
        }
    }
    
    /// <summary>
    /// Get status of a specific Myo or all Myos
    /// </summary>
    public void GetMyoStatus(string address = "")
    {
        if (IsConnected)
        {
            if (string.IsNullOrEmpty(address))
            {
                QueueMessage("GET_MYO_STATUS");
            }
            else
            {
                QueueMessage($"GET_MYO_STATUS,{address}");
            }
            Debug.Log("[NetworkBridge] Requesting Myo status...");
        }
    }
    
    /// <summary>
    /// Send vibration command to a Myo armband (for identification)
    /// </summary>
    public void VibrateDevice(string address)
    {
        if (IsConnected && !string.IsNullOrEmpty(address))
        {
            QueueMessage($"VIBRATE_MYO,{address}");
            Debug.Log($"[NetworkBridge] Sending vibration to: {address}");
        }
    }

    private void ProcessIncoming(string chunk)
    {
        // Prepend any leftover data from previous chunk and parse complete lines
        chunk = leftover + chunk;
        int newlineIdx;
        int startIdx = 0;
        while ((newlineIdx = chunk.IndexOf('\n', startIdx)) != -1)
        {
            string rawLine = chunk.Substring(startIdx, newlineIdx - startIdx).Trim('\r');
            startIdx = newlineIdx + 1;
            if (rawLine.Length == 0) continue; // skip empty
            HandleLine(rawLine);
        }
        // Whatever is left is an incomplete line; store for next chunk
        leftover = chunk.Substring(startIdx);
    }

    private void HandleLine(string line)
    {
        if (line.StartsWith("PREDICTION "))
        {
            var parts = line.Split(' ');
            if (parts.Length >= 3 && float.TryParse(parts[2], out float confidence))
            {
                string pose = parts[1];
                RunOnMainThread(() => PredictionReceived?.Invoke(pose, confidence));
                Debug.Log($"[NetworkBridge] Prediction: {pose} ({confidence:F3})");
            }
            else
            {
                Debug.LogWarning($"[NetworkBridge] Invalid prediction format: {line}");
            }
        }
        else if (line.StartsWith("PREDICTION_DATA "))
        {
            string jsonData = line.Substring(16);
            RunOnMainThread(() => PredictionDataReceived?.Invoke(jsonData));
            Debug.Log($"[NetworkBridge] Prediction data received: {jsonData.Length} chars");
        }
        else if (line.StartsWith("PIPELINE_STATUS "))
        {
            string jsonData = line.Substring(16);
            RunOnMainThread(() => PipelineStatusReceived?.Invoke(jsonData));
            Debug.Log($"[NetworkBridge] Pipeline status: {jsonData}");
        }
        else if (line.StartsWith("PIPELINE "))
        {
            string jsonData = line.Substring(9);
            RunOnMainThread(() => PipelineEventReceived?.Invoke(jsonData));
            Debug.Log($"[NetworkBridge] Pipeline event: {jsonData}");
        }
        else if (line.StartsWith("MODELS "))
        {
            string modelList = line.Substring(7);
            RunOnMainThread(() => ModelListReceived?.Invoke(modelList));
            Debug.Log($"[NetworkBridge] Model list: {modelList}");
        }
        else if (line.StartsWith("ACK MODEL_SET "))
        {
            string modelName = line.Substring(14);
            RunOnMainThread(() => ModelSetConfirmed?.Invoke(modelName));
            Debug.Log($"[NetworkBridge] Model set confirmed: {modelName}");
        }
        else if (line.StartsWith("APRILTAG_STATUS "))
        {
            string statusJson = line.Substring(16);
            RunOnMainThread(() => AprilTagStatusReceived?.Invoke(statusJson));
            Debug.Log($"[NetworkBridge] AprilTag status: {statusJson}");
        }
        else if (line.StartsWith("JOINT_POSITIONS "))
        {
            string jsonData = line.Substring(16);
            RunOnMainThread(() => PredictionDataReceived?.Invoke(jsonData));
        }
        else if (line.StartsWith("ERROR "))
        {
            string error = line.Substring(6);
            RunOnMainThread(() => ErrorReceived?.Invoke(error));
            Debug.LogError($"[NetworkBridge] Server error: {error}");
            messageQueue.Enqueue($"Error: {error}");
        }
        else if (line.StartsWith("MYO_DEVICES "))
        {
            string deviceList = line.Substring(12);
            RunOnMainThread(() => MyoDevicesReceived?.Invoke(deviceList));
            Debug.Log($"[NetworkBridge] Myo devices: {deviceList}");
            messageQueue.Enqueue($"Myo devices found: {deviceList}");
        }
        else if (line.StartsWith("MYO_CONNECTED "))
        {
            string connInfo = line.Substring(14);
            RunOnMainThread(() => MyoConnectedReceived?.Invoke(connInfo));
            Debug.Log($"[NetworkBridge] Myo connected: {connInfo}");
            messageQueue.Enqueue($"Myo connected: {connInfo}");
        }
        else if (line.StartsWith("MYO_DISCONNECTED "))
        {
            string address = line.Substring(17);
            RunOnMainThread(() => MyoDisconnectedReceived?.Invoke(address));
            Debug.Log($"[NetworkBridge] Myo disconnected: {address}");
            messageQueue.Enqueue($"Myo disconnected: {address}");
        }
        else if (line.StartsWith("MYO_STATUS "))
        {
            string statusJson = line.Substring(11);
            RunOnMainThread(() => MyoStatusReceived?.Invoke(statusJson));
            Debug.Log($"[NetworkBridge] Myo status: {statusJson}");
        }
        else if (line.StartsWith("MYO_DEVICES_LIST "))
        {
            string devicesJson = line.Substring(17);
            RunOnMainThread(() => MyoDevicesListReceived?.Invoke(devicesJson));
            Debug.Log($"[NetworkBridge] Myo devices list: {devicesJson}");
        }
        else if (line.StartsWith("MYO_ERROR "))
        {
            string error = line.Substring(10);
            RunOnMainThread(() => MyoErrorReceived?.Invoke(error));
            Debug.LogError($"[NetworkBridge] Myo error: {error}");
            messageQueue.Enqueue($"Myo error: {error}");
        }
        else if (line.StartsWith("ACK"))
        {
            messageQueue.Enqueue(line);
            Debug.Log($"[NetworkBridge] Server ACK: {line}");
        }
        else
        {
            messageQueue.Enqueue(line);
        }
    }
}