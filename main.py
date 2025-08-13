import psutil
import subprocess
import time
import requests
import os
import sys
from vlc_controller import VLCController

class VLCMonitor:
    def __init__(self):
        self.vlc_controller = VLCController()
        self.vlc_http_url = "http://localhost:8080/requests/status.json"
        self.vlc_http_auth = ("", "vlcpassword")
        self.gesture_process = None
        self.monitoring = False
        
    def is_vlc_running(self):
        """Check if VLC process is running"""
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] and "vlc" in proc.info['name'].lower():
                return True
        return False
    
    def is_vlc_active(self):
        """Check if VLC is in a playable state (playing or paused)"""
        try:
            response = requests.get(self.vlc_http_url, auth=self.vlc_http_auth, timeout=2)
            if response.status_code == 200:
                data = response.json()
                state = data.get('state', '')
                # Consider both playing and paused as active
                return state in ['playing', 'paused']
        except:
            # Fallback: if we can't connect, assume active if VLC is running
            return self.is_vlc_running()
        return False
    
    def get_user_confirmation(self):
        """Get user confirmation to start gesture control"""
        print("\n" + "="*50)
        print("🎬 VLC PLAYBACK DETECTED!")
        print("="*50)
        print("A video is currently playing in VLC.")
        print("Would you like to enable gesture control?")
        print("\nAvailable gestures:")
        print("• Show PALM → FIST: Play/Pause toggle")
        print("• Show PALM → TWO FINGERS: Mute/Unmute toggle") 
        print("• Show PALM → FINGER UP: Volume up (continuous)")
        print("• Show PALM → FINGER DOWN: Volume down (continuous)")
        print("• Show PALM again to stop continuous actions")
        print("\n⚠️  This will activate your webcam for gesture recognition.")
        print("="*50)
        
        while True:
            choice = input("Enable gesture control? (y/n): ").lower().strip()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            print("Please enter 'y' for yes or 'n' for no.")
    
    def start_gesture_control(self):
        """Start the gesture control subprocess"""
        if self.gesture_process is None:
            try:
                print("🎥 Starting webcam and gesture recognition...")
                print("Press ESC in the gesture window to stop.")
                
                # Get script path
                base_dir = os.path.dirname(os.path.abspath(__file__))
                gesture_script = os.path.join(base_dir, "gesture_vlc.py")
                
                if not os.path.exists(gesture_script):
                    raise FileNotFoundError("gesture_vlc.py not found")
                
                self.gesture_process = subprocess.Popen(
                    [sys.executable, gesture_script]
                )
                return True
            except Exception as e:
                print(f"❌ Error starting gesture control: {e}")
                return False
        return True
    
    def stop_gesture_control(self):
        """Stop the gesture control subprocess"""
        if self.gesture_process:
            try:
                self.gesture_process.terminate()
                self.gesture_process.wait(timeout=3)
            except:
                try:
                    self.gesture_process.kill()
                except:
                    pass
            finally:
                self.gesture_process = None
                print("📷 Gesture control stopped")
    
    def monitor_playback(self):
        """Monitor VLC playback state and manage gesture control"""
        self.monitoring = True
        was_active = False
        gesture_active = False
        
        print("🔍 Monitoring VLC for media session...")
        print("Press Ctrl+C to exit")
        
        try:
            while self.monitoring:
                vlc_running = self.is_vlc_running()
                currently_active = self.is_vlc_active() if vlc_running else False
                
                # If VLC stops running, clean up
                if not vlc_running:
                    if gesture_active:
                        self.stop_gesture_control()
                        gesture_active = False
                    print("VLC stopped running. Monitoring ended.")
                    break
                
                # If media session just started
                if currently_active and not was_active:
                    if not gesture_active and self.get_user_confirmation():
                        if self.start_gesture_control():
                            gesture_active = True
                            print("✅ Gesture control activated")
                
                # If media session ended
                if not currently_active and was_active:
                    if gesture_active:
                        self.stop_gesture_control()
                        gesture_active = False
                        print("⏹️  Media session ended - gesture control deactivated")
                
                was_active = currently_active
                time.sleep(2)  # Check every 2 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        finally:
            if gesture_active:
                self.stop_gesture_control()
            self.monitoring = False

def main():
    monitor = VLCMonitor()
    
    print("="*40)
    print("VLC Gesture Control System")
    print("="*40)
    
    if not monitor.is_vlc_running():
        print("❌ VLC is not running")
        print("Please start VLC first, then run this program")
        return
    
    print("✅ VLC detected")
    
    # Check HTTP interface
    try:
        response = requests.get(monitor.vlc_http_url, auth=monitor.vlc_http_auth, timeout=2)
        if response.status_code == 200:
            print("✅ VLC HTTP interface accessible")
        else:
            print("⚠️  VLC HTTP interface not responding")
    except:
        print("⚠️  VLC HTTP interface not accessible")
        print("Note: Start VLC with '--intf http --http-password vlcpassword'")
    
    # Start monitoring
    monitor.monitor_playback()

if __name__ == "__main__":
    main()