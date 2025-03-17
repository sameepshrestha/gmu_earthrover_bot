import threading
import requests
import time 
BASE_URL = "http://localhost:8000"
class BotController:
    def __init__(self, base_url):
        self.base_url = base_url
        self.current_speed = 0
        self.current_orientation = 0
        self.current_angle = 0
        self.command_lock = threading.Lock()
        self.running = True
        self.command_thread = threading.Thread(target=self._command_loop)
        self.command_thread.daemon = True
        self.command_thread.start()
        
    def send_control_command(self, linear_speed, angular_value):
        with self.command_lock:
            if linear_speed is None or angular_value is None:
                self.current_speed = 0
                self.current_angle = 0
            else:
                self.current_speed = float(linear_speed)
                self.current_angle = float(angular_value)
    
    def _send_command(self):
        try:
            command = {
                "command": {
                    "linear": self.current_speed,
                    "angular": self.current_angle
                }
            }
            response = requests.post(
                f"{self.base_url}/control",
                json=command,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            # print(f"Command sent - Speed: {self.current_speed:.2f}, Angular: {self.current_angle:.2f}")
        except Exception as e:
            print(f"Error sending control command: {e}")

    def _command_loop(self):
        while self.running:
            with self.command_lock:
                self._send_command()
            time.sleep(0.1)  # Send commands at 10Hz

    def stop(self):
        """Stop the bot and the command loop"""
        self.send_control_command(0, 0)
        self.running = False
        self.command_thread.join()