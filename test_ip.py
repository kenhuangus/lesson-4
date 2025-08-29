import re

# Test IP addresses
test_ips = ["192.168.1.10", "203.0.113.45", "198.51.100.2"]

# Current pattern
pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

print("Testing IP regex pattern:")
print(f"Pattern: {pattern}")
print()

for ip in test_ips:
    match = re.search(pattern, ip)
    print(f"IP: {ip}")
    print(f"  Match: {bool(match)}")
    if match:
        print(f"  Groups: {match.groups()}")
    print()

# Try a simpler pattern
simple_pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
print(f"Simple pattern: {simple_pattern}")
print()

for ip in test_ips:
    match = re.search(simple_pattern, ip)
    print(f"IP: {ip}")
    print(f"  Match: {bool(match)}")
    if match:
        print(f"  Groups: {match.groups()}")
    print()
