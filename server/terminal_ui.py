#!/usr/bin/env python3
"""
Terminal-based UI for Microservice Manager
Works directly in SSH sessions without needing a web browser
"""

import curses
import time
import subprocess
import requests
import psutil
from pathlib import Path
from datetime import datetime

class TerminalUI:
    def __init__(self):
        self.services = []
        self.selected = 0
        self.manager_process = None
        self.message = ""
        
    def get_service_status(self):
        """Get status of all services"""
        services = []
        for i in range(4):
            api_port = 8080 + i
            carla_port = 2000 + (i * 4)
            
            service = {
                'id': i,
                'api_port': api_port,
                'carla_port': carla_port,
                'gpu_id': i % 2,
                'healthy': False,
                'status': 'stopped'
            }
            
            # Check if service is running
            try:
                response = requests.get(f"http://localhost:{api_port}/health", timeout=0.5)
                if response.status_code == 200:
                    service['healthy'] = True
                    service['status'] = 'running'
            except:
                pass
            
            services.append(service)
        return services
    
    def kill_carla_processes(self):
        """Kill all CARLA-related processes"""
        patterns = ["CarlaUE4", "carla_server.py", "microservice_manager"]
        for pattern in patterns:
            subprocess.run(f"pkill -f {pattern}", shell=True, capture_output=True)
        time.sleep(2)
    
    def start_all_services(self):
        """Start all services"""
        if self.manager_process is None or self.manager_process.poll() is not None:
            self.manager_process = subprocess.Popen([
                'python', 'robust_microservice_manager.py',
                '--num-services', '4'
            ])
            self.message = "Starting all services..."
        else:
            self.message = "Services already running"
    
    def stop_all_services(self):
        """Stop all services"""
        if self.manager_process and self.manager_process.poll() is None:
            self.manager_process.terminate()
            time.sleep(2)
            if self.manager_process.poll() is None:
                self.manager_process.kill()
            self.manager_process = None
        self.kill_carla_processes()
        self.message = "All services stopped"
    
    def restart_service(self, service_id):
        """Restart a specific service"""
        api_port = 8080 + service_id
        try:
            requests.post(f"http://localhost:{api_port}/restart", timeout=1)
            self.message = f"Restarting service {service_id}"
        except:
            self.message = f"Failed to restart service {service_id}"
    
    def draw_header(self, stdscr, h, w):
        """Draw header"""
        title = "🚗 Bench2Drive Microservice Manager (Terminal UI)"
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(0, (w - len(title)) // 2, title)
        stdscr.attroff(curses.color_pair(1))
        
        # Draw status line
        timestamp = datetime.now().strftime("%H:%M:%S")
        status = f"[{timestamp}] Services: {len(self.services)} | Healthy: {sum(1 for s in self.services if s['healthy'])}"
        stdscr.addstr(1, 2, status)
        
    def draw_services(self, stdscr, h, w):
        """Draw services list"""
        y_offset = 3
        stdscr.addstr(y_offset, 2, "Services:")
        y_offset += 1
        stdscr.addstr(y_offset, 2, "-" * (w - 4))
        y_offset += 1
        
        for i, service in enumerate(self.services):
            # Highlight selected
            if i == self.selected:
                stdscr.attron(curses.A_REVERSE)
            
            # Choose color based on status
            if service['healthy']:
                stdscr.attron(curses.color_pair(2))  # Green
                status_char = "✓"
            else:
                stdscr.attron(curses.color_pair(3))  # Red
                status_char = "✗"
            
            line = f" [{status_char}] Service {service['id']} | API:{service['api_port']} | CARLA:{service['carla_port']} | GPU:{service['gpu_id']} | {service['status']:8}"
            
            if y_offset < h - 6:
                stdscr.addstr(y_offset, 2, line[:w-4])
            
            stdscr.attroff(curses.color_pair(2))
            stdscr.attroff(curses.color_pair(3))
            if i == self.selected:
                stdscr.attroff(curses.A_REVERSE)
            
            y_offset += 1
        
        return y_offset
    
    def draw_controls(self, stdscr, h, w, y_offset):
        """Draw control instructions"""
        y_offset += 1
        stdscr.addstr(y_offset, 2, "-" * (w - 4))
        y_offset += 1
        
        controls = [
            "Controls:",
            "[↑/↓] Navigate | [Enter] Restart Service | [S] Start All | [X] Stop All",
            "[R] Restart All | [C] Clear CARLA | [Q] Quit"
        ]
        
        for control in controls:
            if y_offset < h - 2:
                stdscr.addstr(y_offset, 2, control)
                y_offset += 1
    
    def draw_message(self, stdscr, h, w):
        """Draw status message"""
        if self.message:
            stdscr.attron(curses.color_pair(4))
            stdscr.addstr(h - 1, 2, f"[{self.message}]")
            stdscr.attroff(curses.color_pair(4))
    
    def run(self, stdscr):
        """Main UI loop"""
        # Initialize colors if available
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            
            # Setup color pairs with proper background
            curses.init_pair(1, curses.COLOR_CYAN, -1)   # Header
            curses.init_pair(2, curses.COLOR_GREEN, -1)  # Healthy
            curses.init_pair(3, curses.COLOR_RED, -1)    # Unhealthy
            curses.init_pair(4, curses.COLOR_YELLOW, -1) # Message
        
        # Setup curses
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)  # Non-blocking input
        
        # Set background
        stdscr.bkgd(' ', curses.A_NORMAL)
        
        last_update = time.time()
        
        while True:
            # Get terminal size
            h, w = stdscr.getmaxyx()
            
            # Update service status every second
            current_time = time.time()
            if current_time - last_update > 1:
                self.services = self.get_service_status()
                last_update = current_time
            
            # Clear and redraw
            stdscr.clear()
            
            self.draw_header(stdscr, h, w)
            y_offset = self.draw_services(stdscr, h, w)
            self.draw_controls(stdscr, h, w, y_offset)
            self.draw_message(stdscr, h, w)
            
            stdscr.refresh()
            
            # Handle input (non-blocking)
            key = stdscr.getch()
            if key == curses.ERR:
                # No input, sleep a bit
                time.sleep(0.1)
                continue
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == curses.KEY_UP:
                self.selected = max(0, self.selected - 1)
                self.message = ""
            elif key == curses.KEY_DOWN:
                self.selected = min(len(self.services) - 1, self.selected + 1)
                self.message = ""
            elif key == ord('\n'):  # Enter - restart selected service
                if self.services:
                    self.restart_service(self.services[self.selected]['id'])
            elif key == ord('s') or key == ord('S'):
                self.start_all_services()
            elif key == ord('x') or key == ord('X'):
                self.stop_all_services()
            elif key == ord('r') or key == ord('R'):
                self.stop_all_services()
                time.sleep(2)
                self.start_all_services()
            elif key == ord('c') or key == ord('C'):
                self.kill_carla_processes()
                self.message = "CARLA processes cleared"

def main():
    """Main entry point"""
    import sys
    
    # Check if we're in an interactive terminal
    if not sys.stdin.isatty():
        print("Error: Terminal UI requires an interactive terminal session.")
        print("Please run directly in your SSH session, not through pipes or scripts.")
        print("\nAlternatives:")
        print("  python monitor.py     # Colored monitor (recommended)")
        print("  python status.py      # Simple status tool")
        sys.exit(1)
    
    try:
        ui = TerminalUI()
        curses.wrapper(ui.run)
    except Exception as e:
        print(f"Error initializing terminal UI: {e}")
        print("\nTry using the colored monitor instead:")
        print("  python monitor.py")
        sys.exit(1)

if __name__ == '__main__':
    main()