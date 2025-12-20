import socket
import json
import time

HOST = '127.0.0.1'
PORT = 8080

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # Disabilita buffer Nagle
    s.settimeout(5.0)
    try:
        print("⏳ Connessione...")
        s.connect((HOST, PORT))
        print("✅ Connesso! (Modalità RAW)")
    except:
        print("❌ Impossibile connettersi.")
        return

    reader = s.makefile('r', encoding='utf-8')
    
    # 1. Movimento AVANTI per 200 frame (ca. 3 secondi)
    print("▶ Invio: AVANTI (P1 e P2)")
    move_action = {"p1_move": "F", "p1_btn": "", "p2_move": "F", "p2_btn": "", "reset": False}
    msg_move = json.dumps(move_action) + "\n"
    
    # 2. Movimento COLPI per 50 frame
    hit_action = {"p1_move": "", "p1_btn": "a", "p2_move": "", "p2_btn": "b", "reset": False}
    msg_hit = json.dumps(hit_action) + "\n"

    try:
        frame = 0
        while True:
            # A. LEGGI STATO (Aspetta il \n da Go)
            line = reader.readline()
            if not line:
                print("❌ Server disconnesso.")
                break
            
            # (Opzionale: stampa ogni 60 frame per vedere che scorre)
            if frame % 60 == 0:
                print(f"Frame {frame}: Sync OK")

            # B. SCEGLI AZIONE
            msg_to_send = msg_move
            if frame > 200: msg_to_send = msg_hit
            if frame > 250: break # Fine test

            # C. INVIA AZIONE (Subito!)
            s.sendall(msg_to_send.encode('utf-8'))
            frame += 1

    except KeyboardInterrupt:
        print("Stop manuale.")
    finally:
        s.close()
        print("Chiuso.")

if __name__ == "__main__":
    main()
