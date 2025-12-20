import socket
import json
import time

HOST = '127.0.0.1'
PORT = 8080

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # Fondamentale
    
    try:
        s.connect((HOST, PORT))
        print("✅ CONNESSO AL SERVER GO")
    except:
        print("❌ Server non trovato")
        return

    reader = s.makefile('r', encoding='utf-8') # Legge righe
    
    # Test: Camminare Avanti per 120 frame
    print("▶ Invio comando: AVANTI per 2 secondi...")
    
    action = {"p1_move": "F", "p1_btn": "", "p2_move": "F", "p2_btn": "", "reset": False}
    msg = json.dumps(action) + "\n" # <--- NOTA IL \n

    try:
        for i in range(120):
            # 1. Aspetta Stato (Sync)
            print(f"Iter: {i}/120")
            # line = reader.readline()
            # if not line: 
            #     print("Empty")
            #     break
            # else:
            #     print(f"Line recieved: {line}")
            
            # 2. Invia Azione
            s.sendall(msg.encode('utf-8'))
            
            if i % 10 == 0: print(f"Frame {i} inviato...")

    except KeyboardInterrupt:
        print("Program killed")

    print("Test Finito.")
    s.close()

if __name__ == "__main__":
    main()
