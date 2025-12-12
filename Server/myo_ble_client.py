"""myo_ble_client.py
BLE Client for Myo Armband connection on server side.


Features:
- Auto-discovery of Myo armbands
- Multi-armband support (track by MAC address)
- EMG data streaming (200 Hz, 16 channels)
- IMU data streaming (quaternion, accel, gyro)
- Battery level monitoring
- Auto-reconnect on disconnect
- Thread-safe callbacks


Usage:
    client = MyoBleclient()
    await client.connect()
    client.set_emg_callback(my_emg_handler)
    await client.start_streaming()
"""


import asyncio
import struct
import time
from typing import Callable, Optional, List, Dict
from bleak import BleakClient, BleakScanner, BleakError
from bleak.backends.device import BLEDevice




# ==================== MYO BLE PROTOCOL ====================


# Service and Characteristic UUIDs (from Thalmic Labs specification)
MYO_SERVICE_UUID = "d5060001-a904-deb9-4748-2c7f4a124842"
COMMAND_CHAR_UUID = "d5060401-a904-deb9-4748-2c7f4a124842"
BATTERY_CHAR_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
EMG_CHAR_UUIDS = [
    "d5060105-a904-deb9-4748-2c7f4a124842",  # EMG0
    "d5060205-a904-deb9-4748-2c7f4a124842",  # EMG1
    "d5060305-a904-deb9-4748-2c7f4a124842",  # EMG2
    "d5060405-a904-deb9-4748-2c7f4a124842",  # EMG3
]
IMU_DATA_CHAR_UUID = "d5060402-a904-deb9-4748-2c7f4a124842"


# Command bytes
COMMAND_SET_MODE = 0x01
COMMAND_VIBRATE = 0x03
COMMAND_SLEEP_MODE = 0x09
COMMAND_UNLOCK = 0x0A


# Command parameters
VIBRATE_SHORT = 0x01
VIBRATE_MEDIUM = 0x02
VIBRATE_LONG = 0x03
SLEEP_NEVER = 0x01
UNLOCK_HOLD = 0x02
EMG_MODE_OFF = 0x00
EMG_MODE_RAW = 0x03
IMU_MODE_OFF = 0x00
IMU_MODE_DATA = 0x01
CLASSIFIER_OFF = 0x00




# ==================== MYO BLE CLIENT ====================


class MyoBleClient:
    """
    BLE client for Myo Armband with server-side features.
   
    Handles connection, streaming, and multi-armband support.
    """
   
    def __init__(self, address: Optional[str] = None, name: Optional[str] = None):
        """
        Initialize Myo BLE client.
       
        Args:
            address: MAC address or device UUID (macOS)
            name: Device name (used if address not provided)
        """
        self.address = address
        self.name = name or "Myo"
        self.device: Optional[BLEDevice] = None
        self.client: Optional[BleakClient] = None
       
        # Connection state
        self.is_connected = False
        self.is_streaming = False
       
        # Callbacks
        self.emg_callback: Optional[Callable] = None
        self.imu_callback: Optional[Callable] = None
        self.battery_callback: Optional[Callable] = None
        self.disconnect_callback: Optional[Callable] = None
       
        # Data buffers
        self.last_battery_level: Optional[int] = None
        self.last_rssi: Optional[int] = None
       
        # Reconnection
        self.auto_reconnect = True
        self.reconnect_task: Optional[asyncio.Task] = None
       
    # ==================== DISCOVERY ====================
   
    @staticmethod
    async def discover_devices(timeout: float = 5.0) -> List[Dict]:
        """
        Discover all Myo armbands in range.
       
        Args:
            timeout: Scan timeout in seconds
           
        Returns:
            List of dicts with device info: {'address', 'name', 'rssi'}
        """
        print(f"[MYO] Scanning for Myo armbands (timeout: {timeout}s)...")
       
        try:
            devices = await BleakScanner.discover(timeout=timeout, service_uuids=[MYO_SERVICE_UUID])
        except Exception as e:
            print(f"[MYO] Scan error: {e}")
            return []
       
        myo_devices = []
        for device in devices:
            # Filter for Myo devices
            if device.name and "myo" in device.name.lower():
                rssi_value = device.rssi if hasattr(device, 'rssi') else None
                info = {
                    'address': device.address,
                    'name': device.name,
                    'rssi': rssi_value
                }
                myo_devices.append(info)
                rssi_str = f"{rssi_value}" if rssi_value is not None else "N/A"
                print(f"[MYO] Found: {device.name} ({device.address}) RSSI: {rssi_str}")
       
        # Also include devices found by service UUID even if name doesn't match
        if not myo_devices and devices:
            print("[MYO] Found devices by service UUID (name filter didn't match)")
            for device in devices:
                rssi_value = device.rssi if hasattr(device, 'rssi') else None
                info = {
                    'address': device.address,
                    'name': device.name or "Unknown Myo",
                    'rssi': rssi_value
                }
                myo_devices.append(info)
                rssi_str = f"{rssi_value}" if rssi_value is not None else "N/A"
                print(f"[MYO] Found: {info['name']} ({device.address}) RSSI: {rssi_str}")
       
        print(f"[MYO] Discovery complete: {len(myo_devices)} device(s) found")
        return myo_devices
   
    # ==================== CONNECTION ====================
   
    async def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to Myo armband.
       
        Args:
            timeout: Connection timeout in seconds
           
        Returns:
            True if connected successfully
        """
        # If no address provided, discover first
        if not self.address:
            print("[MYO] No address specified, discovering...")
            devices = await self.discover_devices(timeout=5.0)
            if not devices:
                print("[MYO] No Myo armbands found")
                return False
           
            # Take first device
            self.address = devices[0]['address']
            self.name = devices[0]['name']
            print(f"[MYO] Selected: {self.name} ({self.address})")
       
        try:
            print(f"[MYO] Connecting to {self.name} ({self.address})...")
           
            # Create client
            self.client = BleakClient(
                self.address,
                disconnected_callback=self._on_disconnect
            )
           
            # Connect with timeout
            await asyncio.wait_for(self.client.connect(), timeout=timeout)
           
            if not self.client.is_connected:
                print("[MYO] Connection failed (not connected)")
                return False
           
            self.is_connected = True
            print(f"[MYO] Connected to {self.name}")
           
            # Initialize Myo
            await self._initialize_myo()
           
            return True
           
        except asyncio.TimeoutError:
            print(f"[MYO] Connection timeout after {timeout}s")
            return False
        except BleakError as e:
            print(f"[MYO] BleakError during connection: {e}")
            return False
        except Exception as e:
            print(f"[MYO] Unexpected error during connection: {e}")
            return False
   
    async def disconnect(self):
        """Disconnect from Myo armband."""
        if not self.client:
            return
       
        try:
            self.auto_reconnect = False  # Disable auto-reconnect
           
            # Stop streaming first
            if self.is_streaming:
                await self.stop_streaming()
           
            # Disconnect
            if self.client.is_connected:
                print(f"[MYO] Disconnecting from {self.name}...")
                await self.client.disconnect()
               
            self.is_connected = False
            print(f"[MYO] Disconnected from {self.name}")
           
        except Exception as e:
            print(f"[MYO] Error during disconnect: {e}")
        finally:
            self.client = None
   
    def _on_disconnect(self, client: BleakClient):
        """
        Callback when device disconnects unexpectedly.
       
        Args:
            client: The BleakClient that disconnected
        """
        self.is_connected = False
        self.is_streaming = False
       
        print(f"[MYO] {self.name} disconnected unexpectedly")
       
        # Call user disconnect callback
        if self.disconnect_callback:
            try:
                self.disconnect_callback(self)
            except Exception as e:
                print(f"[MYO] Error in disconnect callback: {e}")
       
        # Auto-reconnect if enabled
        if self.auto_reconnect:
            print(f"[MYO] Will attempt to reconnect to {self.name}...")
            # Schedule reconnection attempt
            if not self.reconnect_task or self.reconnect_task.done():
                self.reconnect_task = asyncio.create_task(self._reconnect_loop())
   
    async def _reconnect_loop(self):
        """Attempt to reconnect with exponential backoff."""
        retry_delays = [2, 5, 10, 20, 30]  # Backoff delays in seconds
        retry_count = 0
       
        while self.auto_reconnect and not self.is_connected:
            delay = retry_delays[min(retry_count, len(retry_delays) - 1)]
            print(f"[MYO] Reconnection attempt in {delay}s... (attempt {retry_count + 1})")
           
            await asyncio.sleep(delay)
           
            try:
                if await self.connect(timeout=10.0):
                    print(f"[MYO] Reconnected to {self.name} successfully!")
                   
                    # Restart streaming if it was active
                    if self.emg_callback or self.imu_callback:
                        await self.start_streaming(
                            emg=self.emg_callback is not None,
                            imu=self.imu_callback is not None
                        )
                    break
                else:
                    retry_count += 1
            except Exception as e:
                print(f"[MYO] Reconnection attempt failed: {e}")
                retry_count += 1
   
    # ==================== INITIALIZATION ====================
   
    async def _initialize_myo(self):
        """Send initialization commands to Myo after connection."""
        if not self.client or not self.client.is_connected:
            return
       
        try:
            # 1. Unlock Myo (hold mode)
            print("[MYO] Unlocking armband...")
            cmd = struct.pack('<3B', COMMAND_UNLOCK, 1, UNLOCK_HOLD)
            await self.client.write_gatt_char(COMMAND_CHAR_UUID, cmd, response=True)
            await asyncio.sleep(0.1)
           
            # 2. Disable sleep mode
            print("[MYO] Disabling sleep mode...")
            cmd = struct.pack('<3B', COMMAND_SLEEP_MODE, 1, SLEEP_NEVER)
            await self.client.write_gatt_char(COMMAND_CHAR_UUID, cmd, response=True)
            await asyncio.sleep(0.1)
           
            print("[MYO] Initialization complete")
           
        except Exception as e:
            print(f"[MYO] Error during initialization: {e}")
            raise
   
    # ==================== STREAMING ====================
   
    async def start_streaming(self, emg: bool = True, imu: bool = False):
        """
        Start EMG and/or IMU data streaming.
       
        Args:
            emg: Enable EMG streaming (200 Hz)
            imu: Enable IMU streaming
        """
        if not self.client or not self.client.is_connected:
            print("[MYO] Cannot start streaming: not connected")
            return
       
        if self.is_streaming:
            print("[MYO] Already streaming")
            return
       
        try:
            # Set EMG/IMU modes
            emg_mode = EMG_MODE_RAW if emg else EMG_MODE_OFF
            imu_mode = IMU_MODE_DATA if imu else IMU_MODE_OFF
           
            print(f"[MYO] Enabling data streaming (EMG: {emg}, IMU: {imu})...")
            cmd = struct.pack('<5B', COMMAND_SET_MODE, 3, emg_mode, imu_mode, CLASSIFIER_OFF)
            await self.client.write_gatt_char(COMMAND_CHAR_UUID, cmd, response=True)
            await asyncio.sleep(0.2)
           
            # Subscribe to EMG characteristics
            if emg:
                for uuid in EMG_CHAR_UUIDS:
                    await self.client.start_notify(uuid, self._emg_notification_handler)
                    await asyncio.sleep(0.05)  # Small delay between subscriptions
                print("[MYO] EMG notifications enabled")
           
            # Subscribe to IMU characteristic
            if imu:
                await self.client.start_notify(IMU_DATA_CHAR_UUID, self._imu_notification_handler)
                print("[MYO] IMU notifications enabled")
           
            self.is_streaming = True
            print("[MYO] Streaming started")
           
        except Exception as e:
            print(f"[MYO] Error starting streaming: {e}")
            raise
   
    async def stop_streaming(self):
        """Stop EMG and IMU data streaming."""
        if not self.client or not self.client.is_connected:
            return
       
        if not self.is_streaming:
            return
       
        try:
            print("[MYO] Stopping data streaming...")
           
            # Disable EMG/IMU modes
            cmd = struct.pack('<5B', COMMAND_SET_MODE, 3, EMG_MODE_OFF, IMU_MODE_OFF, CLASSIFIER_OFF)
            await self.client.write_gatt_char(COMMAND_CHAR_UUID, cmd, response=True)
           
            # Unsubscribe from notifications
            for uuid in EMG_CHAR_UUIDS:
                try:
                    await self.client.stop_notify(uuid)
                except:
                    pass
           
            try:
                await self.client.stop_notify(IMU_DATA_CHAR_UUID)
            except:
                pass
           
            self.is_streaming = False
            print("[MYO] Streaming stopped")
           
        except Exception as e:
            print(f"[MYO] Error stopping streaming: {e}")
   
    # ==================== NOTIFICATION HANDLERS ====================
   
    def _emg_notification_handler(self, sender, data: bytearray):
        """
        Handle incoming EMG data notifications.
       
        Each notification contains 16 bytes (two sequential 8-channel frames).
        The callback receives the full 16-byte packet so downstream
        processing retains both readings per timestamp.
       
        Args:
            sender: Characteristic UUID
            data: Raw EMG data (16 signed bytes)
        """
        if not self.emg_callback:
            return
       
        try:
            # Unpack 16 signed bytes (two sequential 8-channel frames)
            emg_readings = struct.unpack('<16b', data)


            # Provide the full 16-byte packet so downstream code retains both frames
            self.emg_callback(emg_readings, self)


        except Exception as e:
            print(f"[MYO] Error in EMG handler: {e}")
   
    def _imu_notification_handler(self, sender, data: bytearray):
        """
        Handle incoming IMU data notifications.
       
        IMU data format (20 bytes):
        - Quaternion (4 × int16): orientation
        - Accelerometer (3 × int16): m/s²
        - Gyroscope (3 × int16): deg/s
       
        Args:
            sender: Characteristic UUID
            data: Raw IMU data
        """
        if not self.imu_callback:
            return
       
        try:
            # Parse IMU data (20 bytes)
            if len(data) >= 20:
                # Quaternion (w, x, y, z) - normalized to range [-1, 1]
                qw, qx, qy, qz = struct.unpack('<4h', data[0:8])
                quaternion = (qw / 16384.0, qx / 16384.0, qy / 16384.0, qz / 16384.0)
               
                # Accelerometer (x, y, z) - in units of g
                ax, ay, az = struct.unpack('<3h', data[8:14])
                accel = (ax / 2048.0, ay / 2048.0, az / 2048.0)
               
                # Gyroscope (x, y, z) - in deg/s
                gx, gy, gz = struct.unpack('<3h', data[14:20])
                gyro = (gx / 16.0, gy / 16.0, gz / 16.0)
               
                # Call user callback
                self.imu_callback(quaternion, accel, gyro, self)
               
        except Exception as e:
            print(f"[MYO] Error in IMU handler: {e}")
   
    # ==================== COMMANDS ====================
   
    async def vibrate(self, duration: str = "short"):
        """
        Send vibration command to armband.
       
        Args:
            duration: "short", "medium", or "long"
        """
        if not self.client or not self.client.is_connected:
            print("[MYO] Cannot vibrate: not connected")
            return
       
        try:
            vibrate_types = {
                "short": VIBRATE_SHORT,
                "medium": VIBRATE_MEDIUM,
                "long": VIBRATE_LONG
            }
           
            vibrate_val = vibrate_types.get(duration, VIBRATE_SHORT)
            cmd = struct.pack('<3B', COMMAND_VIBRATE, 1, vibrate_val)
            await self.client.write_gatt_char(COMMAND_CHAR_UUID, cmd, response=True)
            print(f"[MYO] Vibration sent ({duration})")
           
        except Exception as e:
            print(f"[MYO] Error sending vibration: {e}")
   
    async def read_battery_level(self) -> Optional[int]:
        """
        Read battery level from armband.
       
        Returns:
            Battery level percentage (0-100) or None if failed
        """
        if not self.client or not self.client.is_connected:
            print("[MYO] Cannot read battery: not connected")
            return None
       
        try:
            battery_bytes = await self.client.read_gatt_char(BATTERY_CHAR_UUID)
            battery_level = int.from_bytes(battery_bytes, byteorder='little')
            self.last_battery_level = battery_level
           
            # Call battery callback if set
            if self.battery_callback:
                self.battery_callback(battery_level, self)
           
            return battery_level
           
        except Exception as e:
            print(f"[MYO] Error reading battery: {e}")
            return None
   
    # ==================== CALLBACKS ====================
   
    def set_emg_callback(self, callback: Optional[Callable]):
        """
        Set callback for EMG data.


        Callback signature: callback(emg_sample: tuple[int], client: MyoBleClient)
        - emg_sample: Tuple of 16 signed integers (-128 to 127) representing the two
          sequential 8-channel frames contained in each BLE packet.
        - client: This MyoBleClient instance
        """
        self.emg_callback = callback
   
    def set_imu_callback(self, callback: Optional[Callable]):
        """
        Set callback for IMU data.
       
        Callback signature: callback(quat, accel, gyro, client: MyoBleClient)
        - quat: (w, x, y, z) quaternion
        - accel: (x, y, z) acceleration in g
        - gyro: (x, y, z) angular velocity in deg/s
        - client: This MyoBleClient instance
        """
        self.imu_callback = callback
   
    def set_battery_callback(self, callback: Optional[Callable]):
        """
        Set callback for battery level updates.
       
        Callback signature: callback(level: int, client: MyoBleClient)
        """
        self.battery_callback = callback
   
    def set_disconnect_callback(self, callback: Optional[Callable]):
        """
        Set callback for disconnect events.
       
        Callback signature: callback(client: MyoBleClient)
        """
        self.disconnect_callback = callback
   
    # ==================== STATUS ====================
   
    def get_status(self) -> Dict:
        """
        Get current connection status.
       
        Returns:
            Dictionary with status information
        """
        return {
            'address': self.address,
            'name': self.name,
            'connected': self.is_connected,
            'streaming': self.is_streaming,
            'battery': self.last_battery_level,
            'rssi': self.last_rssi
        }




# ==================== MULTI-ARMBAND MANAGER ====================


class MyoManager:
    """
    Manager for multiple Myo armbands.
   
    Tracks multiple armbands by MAC address.
    """
   
    def __init__(self):
        self.armbands: Dict[str, MyoBleClient] = {}  # {address: client}
        self.lock = asyncio.Lock()
   
    async def discover(self, timeout: float = 5.0) -> List[Dict]:
        """Discover all Myo armbands in range."""
        return await MyoBleClient.discover_devices(timeout=timeout)
   
    async def connect(self, address: str, name: str = "Myo") -> Optional[MyoBleClient]:
        """
        Connect to a Myo armband.
       
        Args:
            address: MAC address or UUID
            name: Device name
           
        Returns:
            MyoBleClient instance or None if failed
        """
        async with self.lock:
            # Check if already connected
            if address in self.armbands:
                client = self.armbands[address]
                if client.is_connected:
                    print(f"[MYO_MANAGER] Already connected to {address}")
                    return client
           
            # Create new client
            client = MyoBleClient(address=address, name=name)
           
            # Connect
            if await client.connect():
                self.armbands[address] = client
                print(f"[MYO_MANAGER] Connected to {name} ({address})")
                return client
            else:
                print(f"[MYO_MANAGER] Failed to connect to {address}")
                return None
   
    async def disconnect(self, address: str):
        """Disconnect from a specific armband."""
        async with self.lock:
            if address in self.armbands:
                client = self.armbands[address]
                await client.disconnect()
                del self.armbands[address]
                print(f"[MYO_MANAGER] Disconnected from {address}")
   
    async def disconnect_all(self):
        """Disconnect from all armbands."""
        async with self.lock:
            for address, client in list(self.armbands.items()):
                await client.disconnect()
            self.armbands.clear()
            print("[MYO_MANAGER] Disconnected from all armbands")
   
    def get_client(self, address: str) -> Optional[MyoBleClient]:
        """Get client for specific armband."""
        return self.armbands.get(address)
   
    def get_all_clients(self) -> List[MyoBleClient]:
        """Get all connected clients."""
        return list(self.armbands.values())
   
    def get_status_all(self) -> Dict[str, Dict]:
        """Get status of all armbands."""
        return {
            address: client.get_status()
            for address, client in self.armbands.items()
        }




# ==================== STANDALONE TEST ====================


async def test_myo_connection():
    """Test script for standalone usage."""
    print("=== Myo BLE Client Test ===\n")
   
    # Discover devices
    devices = await MyoBleClient.discover_devices(timeout=5.0)
    if not devices:
        print("No Myo armbands found!")
        return
   
    # Connect to first device
    device = devices[0]
    client = MyoBleClient(address=device['address'], name=device['name'])
   
    # Set callbacks
    def emg_handler(sample, client):
        print(f"EMG: {sample}")
   
    def battery_handler(level, client):
        print(f"Battery: {level}%")
   
    client.set_emg_callback(emg_handler)
    client.set_battery_callback(battery_handler)
   
    # Connect
    if await client.connect():
        print("\n✅ Connected successfully!\n")
       
        # Read battery
        battery = await client.read_battery_level()
        print(f"Battery: {battery}%\n")
       
        # Start streaming
        await client.start_streaming(emg=True, imu=False)
       
        # Stream for 10 seconds
        print("Streaming EMG data for 10 seconds...")
        await asyncio.sleep(10)
       
        # Stop streaming
        await client.stop_streaming()
       
        # Disconnect
        await client.disconnect()
       
        print("\n✅ Test complete!")
    else:
        print("\n❌ Connection failed!")




if __name__ == "__main__":
    asyncio.run(test_myo_connection())





