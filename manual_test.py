import socket
import json
import time

HOST = '127.0.0.1'
PORT = 8080

# Mappa dei comandi semplici per il test manuale
COMMANDS = {
    "W": {"p1_move": "U", "p1_btn": ""},  # Salto (Up)
    "S": {"p1_move": "D", "p1_btn": ""},  # Abbassati (Down)
    "A": {"p1_move": "B", "p1_btn": ""},  # Indietro (Back)
    "D": {"p1_move": "F", "p1_btn": ""},  # Avanti (Forward)
    "J": {"p1_move": "",  "p1_btn": "a"}, # Pugno Debole (Button A)
    "K": {"p1_move": "",  "p1_btn": "b"}, # Calcio Debole (Button B)
    "N": {"p1_move": "",  "p1_btn": ""},  # Neutro (Nessun input)
    "R": {"p1_move": "",  "p1_btn": "", "reset": True} # RESET
}

def main():
    print("--- IKEMEN GO MANUAL CONTROLLER ---")
    print("1. Avvia IKEMEN GO in un altro terminale.")
    print("2. Il gioco si bloccherà in attesa di questo script.")
    print("---------------------------------------")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Tentativo di connessione
    connected = False
    while not connected:
        try:
            print("Tentativo di connessione a 127.0.0.1:8080...")
            sock.connect((HOST, PORT))
            connected = True
            print("✅ CONNESSO! Il gioco dovrebbe sbloccarsi ora.")
        except ConnectionRefusedError:
            print("❌ Server non trovato. Assicurati che ./Ikemen_RL sia avviato.")
            time.sleep(2)

    try:
        # Invia un primo comando neutro per avviare lo scambio
        initial_action = {"p1_move": "", "p1_btn": "", "reset": False}
        sock.sendall(json.dumps(initial_action).encode('utf-8'))

        while True:
            # 1. RICEZIONE STATO (READ)
            data = sock.recv(4096)
            if not data:
                print("Server disconnesso.")
                break
            
            try:
                state = json.loads(data.decode('utf-8'))
                # Stampa formattata dello stato (Prova che stiamo LEGGENDO la memoria)
                print(f"\n[READ] Tick: {state['tick']} | P1 HP: {state['p1_hp']} | X: {state['p1_x']:.2f} | Facing: {state['p1_facing']}")
            except json.JSONDecodeError:
                print("Errore decodifica JSON (pacchetto parziale?), salto...")
                continue

            # 2. INVIO COMANDO (WRITE)
            print("Comandi: W(Su) A(Indietro) S(Giù) D(Avanti) J(Pugno) K(Calcio) R(Reset) INVIO(Neutro)")
            user_input = input("Azione > ").upper()
            
            action = COMMANDS.get(user_input, COMMANDS["N"]).copy()
            
            # Gestione Reset speciale
            if user_input == "R":
                action["reset"] = True
                print(">>> INVIO RESET...")

            # Invio al server
            sock.sendall(json.dumps(action).encode('utf-8'))

    except KeyboardInterrupt:
        print("\nChiusura test.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
