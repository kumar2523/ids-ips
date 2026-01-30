from scapy.all import IP, TCP, UDP, ICMP, wrpcap
import pandas as pd
import random

# File path for NSL-KDD Test dataset
txt_file = "KDDTest+.txt"

# Read the dataset (tab-separated or comma-separated)
df = pd.read_csv(txt_file, header=None, delimiter=",", low_memory=False)

# Define Kali Linux IP and Destination IP
src_ip = "192.168.48.130"  # Your Kali Linux machine
dst_ip = "192.168.48.1"    # Another machine/router in your network

packets = []

# Iterate through dataset and generate packets
for _, row in df.iterrows():
    protocol = row[1]  # Protocol type (tcp, udp, icmp)
    label = row.iloc[-1]  # Attack type or normal

    # Random source port
    src_port = random.randint(1024, 65535)
    dst_port = 80 if protocol == "tcp" else 53  # HTTP for TCP, DNS for UDP

    # Create packets based on protocol type
    if protocol == "tcp":
        pkt = IP(src=src_ip, dst=dst_ip) / TCP(sport=src_port, dport=dst_port)
    elif protocol == "udp":
        pkt = IP(src=src_ip, dst=dst_ip) / UDP(sport=src_port, dport=dst_port)
    elif protocol == "icmp":
        pkt = IP(src=src_ip, dst=dst_ip) / ICMP()
    else:
        continue  # Skip unknown protocols

    packets.append(pkt)

# Save the generated packets as a PCAP file
pcap_file = "test_traffic.pcap"
wrpcap(pcap_file, packets)

print(f"✅ PCAP file successfully saved as {pcap_file}")