import socket
import json
import time

HOST = '127.0.0.1'
PORT = 8080

# --- CONFIGURAZIONE DELLA SEQUENZA ---
# Ogni elemento è: (DURATA_FRAMES, MOVIMENTO, TASTO)
# Esempio: "Giù" per 30 frame, poi "Avanti" per 10, poi "Pugno(a)" per 5
ATTACK_SEQUENCE = [
    (10, "",   ""),   # Preparazione
    
    # --- Primo Giro (QCF) ---
    (4,  "D",  ""),   
    (4,  "DF", ""),   
    (4,  "F",  "xy"),   # Solo movimento, niente tasto qui
    
    # --- Secondo Giro (QCF) ---
    # Nota: Passare da F subito a D è tecnicamente possibile via script,
    # ma a volte un frame di "DF" o Neutro nel mezzo aiuta la lettura.
    (4,  "D",  ""),   
    (4,  "DF", ""),   
    
    # --- Attivazione ---
    (10, "F",  "y"),  # Qui premi il tasto finale!
    (40, "F",  ""),   # Tieni premuto avanti durante l'animazione della super
] 

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    try:
        s.connect((HOST, PORT))
        print("✅ CONNESSO AL SERVER GO")
    except:
        print("❌ Server non trovato. Avvia prima il gioco!")
        return

    reader = s.makefile('r', encoding='utf-8')

    print(f"▶ Avvio sequenza di {len(ATTACK_SEQUENCE)} passaggi...")

    # Variabili per gestire la sequenza
    seq_index = 0       # A quale passo siamo
    frame_count = 0     # Da quanto tempo siamo in questo passo
    running = True

    try:
        while running:
            # 1. Aspetta lo Stato dal Server (Sync step)
            # Questo è fondamentale: il server scandisce il tempo.
            line = reader.readline()
            if not line: 
                print("Server disconnesso.")
                break
            
            # --- LOGICA SEQUENZA ---
            # Recupera il passo corrente
            duration, move, btn = ATTACK_SEQUENCE[seq_index]

            # Prepara il JSON
            action = {
                "p1_move": move, 
                "p1_btn": btn, 
                "p2_move": "",  # Player 2 fermo
                "p2_btn": "", 
                "reset": False
            }
            
            # Invia
            msg = json.dumps(action) + "\n"
            s.sendall(msg.encode('utf-8'))

            # --- GESTIONE TEMPO ---
            frame_count += 1
            
            # Se abbiamo eseguito questo passo abbastanza a lungo, passa al prossimo
            if frame_count >= duration:
                frame_count = 0
                seq_index += 1
                print(f" -> Passo {seq_index} completato. Next: {ATTACK_SEQUENCE[seq_index % len(ATTACK_SEQUENCE)]}")

                # Loop: Se la sequenza finisce, ricomincia da capo
                if seq_index >= len(ATTACK_SEQUENCE):
                    seq_index = 0
                    print("↺ Sequenza riavviata")

    except KeyboardInterrupt:
        print("\nStop manuale.")
    finally:
        s.close()
        print("Connessione chiusa.")

if __name__ == "__main__":
    main()
