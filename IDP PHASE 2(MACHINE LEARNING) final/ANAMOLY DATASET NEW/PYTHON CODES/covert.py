# this below code is to get pcap file from .csv file
import pandas as pd
from scapy.all import IP, TCP, UDP, Ether, wrpcap

# Define input CSV and output PCAP path
csv_file = r"C:\Users\ivmks\OneDrive\Desktop\nsl_kdd_naivebayes\data\anomaly_traffic_clean.csv"
pcap_file = r"C:\Users\ivmks\OneDrive\Desktop\nsl_kdd_naivebayes\data\anomaly_traffic.pcap"

# Load the cleaned dataset
df = pd.read_csv(csv_file)

# List to store packets
packets = []

# Convert dataset rows into network packets
for _, row in df.iterrows():
    src_ip = "192.168.1.100"  # Simulated source IP
    dst_ip = "192.168.1.200"  # Simulated destination IP
    src_port = 12345
    dst_port = 80 if row["protocol_type"] == "tcp" else 53  # TCP (HTTP) or UDP (DNS)
    
    # Create packet based on protocol
    if row["protocol_type"] == "tcp":
        pkt = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=src_port, dport=dst_port)
    else:
        pkt = Ether() / IP(src=src_ip, dst=dst_ip) / UDP(sport=src_port, dport=dst_port)
    
    packets.append(pkt)

# Save packets to PCAP file
wrpcap(pcap_file, packets)

print(f"✅ PCAP file saved at: {pcap_file}")
