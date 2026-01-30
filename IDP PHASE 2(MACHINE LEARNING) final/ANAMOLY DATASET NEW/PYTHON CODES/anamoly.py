# this below code is for remove signature atacks from nsl-kdd dataset.
# and giving all the anamoly attacks information..
import pandas as pd

# Define dataset paths
data_path = r"C:\Users\ivmks\OneDrive\Desktop\nsl_kdd_naivebayes\data"
train_file = f"{data_path}\\KDDTrain+.txt"
test_file = f"{data_path}\\KDDTest+.txt"
output_file = f"{data_path}\\anomaly_traffic_clean.csv"

# Define column names (based on NSL-KDD dataset format)
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", 
    "logged_in", "num_compromised", "root_shell", "su_attempted", 
    "num_root", "num_file_creations", "num_shells", "num_access_files", 
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count", 
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", 
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
    "dst_host_serror_rate", "dst_host_srv_serror_rate", 
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type"
]

# List of known signature-based attacks to remove
signature_attacks = {
    "ipsweep", "nmap", "portsweep", "satan",  # Probe attacks
    "back", "land", "neptune", "pod", "smurf", "teardrop",  # DoS attacks
    "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster",  # R2L
    "buffer_overflow", "loadmodule", "perl", "rootkit"  # U2R
}

# Load datasets (ignoring the last difficulty_level column)
train_df = pd.read_csv(train_file, header=None, names=columns, usecols=range(len(columns)))
test_df = pd.read_csv(test_file, header=None, names=columns, usecols=range(len(columns)))

# Combine datasets
full_df = pd.concat([train_df, test_df])

# Remove all signature-based attacks
filtered_df = full_df[~full_df["attack_type"].isin(signature_attacks)]

# Save the cleaned dataset
filtered_df.to_csv(output_file, index=False)

print(f"✅ Cleaned anomaly dataset saved at: {output_file}")
