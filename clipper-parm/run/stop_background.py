import argparse
import socket


if __name__ == "__main__":
    sock_name = '/home/ubuntu/bg_sock'
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    print("Connecting to uds")
    sock.connect(sock_name)
    print("Connected to uds. Waiting to receive response")
    sock.recv(1)
    print("Received response. Closing.")
    sock.close()
