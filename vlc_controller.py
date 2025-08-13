import pyautogui
import requests
import time
import json

class VLCController:
    def __init__(self):
        self.vlc_http_url = "http://localhost:8080/requests"
        self.vlc_http_auth = ("", "vlcpassword")
        self.use_http_api = True
        self.volume_step = 5  # Default step for one-time actions
        self.continuous_step = 2  # Smaller step for continuous actions
        
        # Test HTTP API availability
        try:
            response = requests.get(f"{self.vlc_http_url}/status.json", 
                                  auth=self.vlc_http_auth, timeout=1)
            if response.status_code != 200:
                self.use_http_api = False
        except:
            self.use_http_api = False
            
        if not self.use_http_api:
            print("⚠️  VLC HTTP API not available. Using keyboard shortcuts.")
        else:
            print("✅ VLC HTTP API connected.")
    
    def play_pause(self):
        """Toggle play/pause"""
        if self.use_http_api:
            try:
                requests.get(f"{self.vlc_http_url}/status.json?command=pl_pause", 
                           auth=self.vlc_http_auth, timeout=1)
                return True
            except:
                pass
        
        # Fallback to keyboard shortcut
        pyautogui.press('space')
        return True
    
    # def volume_up(self, continuous=False):
    #     """Increase volume with smaller steps for continuous adjustment"""
    #     step = self.continuous_step if continuous else self.volume_step
        
    #     if self.use_http_api:
    #         try:
    #             requests.get(f"{self.vlc_http_url}/status.json?command=volume&val=+{step}", 
    #                        auth=self.vlc_http_auth, timeout=1)
    #             return True
    #         except:
    #             pass
        
    #     # Fallback to keyboard shortcut
    #     pyautogui.press('up')
    #     return True
    
    # def volume_down(self, continuous=False):
    #     """Decrease volume with smaller steps for continuous adjustment"""
    #     step = self.continuous_step if continuous else self.volume_step
        
    #     if self.use_http_api:
    #         try:
    #             requests.get(f"{self.vlc_http_url}/status.json?command=volume&val=-{step}", 
    #                        auth=self.vlc_http_auth, timeout=1)
    #             return True
    #         except:
    #             pass
        
    #     # Fallback to keyboard shortcut
    #     pyautogui.press('down')
    #     return True
    

    def volume_up(self, continuous=False):
        """Increase volume with smaller steps for continuous adjustment"""
        step = 5 if continuous else self.volume_step  # smoother step

        if self.use_http_api:
            try:
                requests.get(f"{self.vlc_http_url}/status.json?command=volume&val=+{step}",
                            auth=self.vlc_http_auth, timeout=1)
                return True
            except:
                pass

        # Fallback to keyboard shortcut (simulate multiple small presses)
        if continuous:
            for _ in range(step):  # simulate smaller increments
                pyautogui.press('up')
                time.sleep(0.02)  # tiny delay for smoothness
        else:
            pyautogui.press('up')
        return True


    def volume_down(self, continuous=False):
        """Decrease volume with smaller steps for continuous adjustment"""
        step = 5 if continuous else self.volume_step  # smoother step

        if self.use_http_api:
            try:
                requests.get(f"{self.vlc_http_url}/status.json?command=volume&val=-{step}",
                            auth=self.vlc_http_auth, timeout=1)
                return True
            except:
                pass

        # Fallback to keyboard shortcut (simulate multiple small presses)
        if continuous:
            for _ in range(step):
                pyautogui.press('down')
                time.sleep(0.02)
        else:
            pyautogui.press('down')
        return True




    def mute_toggle(self):
        """Toggle mute/unmute using VLC HTTP API"""
        if self.use_http_api:
            try:
                # Check current status
                response = requests.get(f"{self.vlc_http_url}/status.json", 
                                        auth=self.vlc_http_auth, timeout=1)
                if response.status_code == 200:
                    data = response.json()
                    is_muted = data.get('volume') == 0  # VLC uses volume=0 when muted
                    
                    if is_muted:
                        # Unmute: set volume back to a default (e.g., 256)
                        requests.get(f"{self.vlc_http_url}/status.json?command=volume&val=256", 
                                    auth=self.vlc_http_auth, timeout=1)
                    else:
                        # Mute: set volume to 0
                        requests.get(f"{self.vlc_http_url}/status.json?command=volume&val=0", 
                                    auth=self.vlc_http_auth, timeout=1)
                    return True
            except:
                pass

        # Fallback to keyboard shortcut
        pyautogui.press('m')
        return True



    def get_volume(self):
        """Get current volume level"""
        if self.use_http_api:
            try:
                response = requests.get(f"{self.vlc_http_url}/status.json", 
                                      auth=self.vlc_http_auth, timeout=1)
                if response.status_code == 200:
                    data = response.json()
                    return data.get('volume', 100)
            except:
                pass
        return None
    
    def get_status(self):
        """Get VLC playback status"""
        if self.use_http_api:
            try:
                response = requests.get(f"{self.vlc_http_url}/status.json", 
                                      auth=self.vlc_http_auth, timeout=1)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'state': data.get('state', 'unknown'),
                        'volume': data.get('volume', 100),
                        'position': data.get('position', 0),
                        'length': data.get('length', 0)
                    }
            except:
                pass
        return None