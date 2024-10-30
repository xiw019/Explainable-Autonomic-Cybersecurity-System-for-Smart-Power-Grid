from scapy.all import rdpcap, IP, TCP, UDP
import csv
import pandas as pd 


def pcap_to_csv(pcap_file, csv_file):
    packets = rdpcap(pcap_file)
    # Adding 'label' to the CSV header
    csv_header = ['timestamp', 'length', 'highest_layer', 'src_port', 'dst_port', 'protocol', 'ttl', 'tcp_flags', 'payload_length', 'label']
    
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        
        start_time = packets[0].time if packets else 0
        attack_start = 91.59370
        attack_end = 120.922708
        
        for packet in packets:
            timestamp = packet.time - start_time
            length = len(packet)
            highest_layer = packet.lastlayer().name
            src_port, dst_port = 'N/A', 'N/A'
            protocol = 'N/A'
            ttl = 'N/A'
            tcp_flags = 'N/A'
            payload_length = 'N/A'
            label = 'normal'  # Default label
            
            if attack_start <= timestamp <= attack_end:
                label = 'attack'
            
            if IP in packet:
                protocol = packet[IP].proto
                ttl = packet[IP].ttl
                
                if TCP in packet:
                    src_port = packet[TCP].sport
                    dst_port = packet[TCP].dport
                    tcp_flags = packet.sprintf("%TCP.flags%")
                    payload_length = len(packet[TCP].payload) if packet[TCP].payload else 0
                elif UDP in packet:
                    src_port = packet[UDP].sport
                    dst_port = packet[UDP].dport
                    payload_length = len(packet[UDP].payload) if packet[UDP].payload else 0
            
            writer.writerow([f"{timestamp:.6f}", length, highest_layer, src_port, dst_port, protocol, ttl, tcp_flags, payload_length, label])

def categorize_port(port):
    if port <= 1023:
        return 'Well-known'
    elif port <= 49151:
        return 'Registered'
    elif port == 20 or port == 21:
        return 'FTP_20_21'
    elif port == 22:
        return 'SSH_22'
    elif port == 25:
        return 'SMTP_25'
    elif port == 53:
        return 'DNS_53'
    elif port == 80:
        return 'HTTP_80'
    elif port == 123:
        return 'NTP_123'
    elif port == 179:
        return 'BGP_179'
    elif port == 443:
        return 'HTTPS_443'
    elif port == 500:
        return 'ISAKMP_500'
    elif port == 587:
        return 'Secure_SMTP_587'
    elif port == 3389:
        return 'RDP_3389'
    else:
        return 'Dynamic_Private'
 
pcap_file = 'syn-flood.pcap'
csv_file = 'syn-flood.csv'
pcap_to_csv(pcap_file, csv_file)

df = pd.read_csv('syn-flood.csv')

df['src_port'] = df['src_port'].apply(categorize_port)
df['dst_port'] = df['dst_port'].apply(categorize_port)

# Save the transformed dataframe to a new CSV file (optional)
df.to_csv('syn-flood.csv', index=False)  