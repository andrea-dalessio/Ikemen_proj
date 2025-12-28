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

    cnt = 0
    try:
        while True:
            # 1. Aspetta Stato (Sync)
            line = reader.readline()
            if not line: 
                print("Empty")
                break
            if cnt == 0:
                msg = json.dumps({"p1_move": "F", "p1_btn": "", "p2_move": "B", "p2_btn": "", "reset": False}) + "\n"
                print("Start behaviour")
                cnt += 1
            if cnt == 360:
                msg = json.dumps({"p1_move": "", "p1_btn": "", "p2_move": "", "p2_btn": "", "reset": True})+'\n'
                print("Reset...", end=' ')
                cnt += 1
            elif cnt == 361:
                msg = json.dumps({"p1_move": "", "p1_btn": "", "p2_move": "", "p2_btn": "", "reset": False})+'\n'
                cnt = 0
                print("Done")
            else:
                cnt += 1
            
            # 2. Invia Azione
            s.sendall(msg.encode('utf-8'))
    except KeyboardInterrupt:
        print("Program killed")
    finally:
        print("Test Finito.")
        s.close()

if __name__ == "__main__":
    main()
